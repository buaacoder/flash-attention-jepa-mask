# Variable_seqlenk_mask功能详细说明

## 概述

Variable_seqlenk_mask是Flash Attention的第4种mask模式，允许每个query位置attention到不同数量的keys。与现有的三种mask模式不同，这种模式通过传递一个二维数组 `per_row_seqlens_k` 来为每个query位置指定可见的key数量。

## 核心特性

- **灵活的attention窗口**: 每个query位置可以有独立的seqlen_k值
- **高效实现**: 在CUDA kernel中直接支持，无需额外的预处理
- **向后兼容**: 当不提供per_row_seqlens_k时，行为与标准attention一致
- **支持梯度**: 完整支持前向和后向传播

## 参数说明

### per_row_seqlens_k
- **类型**: `torch.Tensor`
- **数据类型**: `torch.int32`
- **维度**: `(batch_size, seqlen_q)` - 主要格式，更直观易用
- **兼容维度**: `(batch_size * seqlen_q,)` - 一维格式，向后兼容
- **设备**: 必须与q,k,v在同一设备上
- **描述**: 为每个query位置指定可以attention到的key数量

## 使用方法

### 基础用法

```python
import torch
from flash_attn_interface import flash_attn_func

# 创建输入
batch_size, seqlen_q, seqlen_k = 2, 4, 6
q = torch.randn(batch_size, seqlen_q, 8, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(batch_size, seqlen_k, 8, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(batch_size, seqlen_k, 8, 64, dtype=torch.bfloat16, device='cuda')

# 定义每个query位置的seqlen_k (推荐的二维格式)
per_row_seqlens_k = torch.tensor([
    [3, 4, 2, 5],  # batch 0: query 0->3 keys, query 1->4 keys, etc.
    [6, 1, 3, 4]   # batch 1: query 0->6 keys, query 1->1 keys, etc.
], dtype=torch.int32, device='cuda')

# 执行attention
out, lse = flash_attn_func(q, k, v, per_row_seqlens_k=per_row_seqlens_k)
```

### 兼容的一维格式

```python
# 一维格式 (向后兼容)
per_row_seqlens_k_1d = torch.tensor([3, 4, 2, 5, 6, 1, 3, 4], 
                                   dtype=torch.int32, device='cuda')
out, lse = flash_attn_func(q, k, v, per_row_seqlens_k=per_row_seqlens_k_1d)
```

## 应用场景

### 1. 动态序列长度

处理batch中不同长度的序列，避免padding带来的计算浪费：

```python
# batch中各序列的实际长度
actual_lengths = [3, 5, 2]
batch_size, max_seqlen = 3, 5

# 构造per_row_seqlens_k
per_row_seqlens_k = []
for seq_len in actual_lengths:
    row = [seq_len] * max_seqlen  # 该batch所有query只能看到有效的keys
    per_row_seqlens_k.append(row)

per_row_seqlens_k = torch.tensor(per_row_seqlens_k, dtype=torch.int32, device='cuda')
```

### 2. Progressive Attention

实现逐步增加的attention窗口，常用于训练时的curriculum learning：

```python
seqlen = 6
# position i 可以看到前 i+1 个tokens
progressive_lengths = [i + 1 for i in range(seqlen)]
per_row_seqlens_k = torch.tensor([progressive_lengths], dtype=torch.int32, device='cuda')
```

### 3. 非对称Attention

为不同的query位置分配不同的attention范围：

```python
# 前面的tokens看得少，后面的tokens看得多
seqlen_q = 8
asymmetric_lengths = [2, 3, 4, 5, 6, 7, 8, 8]  # 递增的attention窗口
per_row_seqlens_k = torch.tensor([asymmetric_lengths], dtype=torch.int32, device='cuda')
```

## 实现细节

### C++ API修改

#### flash.h
```cpp
struct Flash_fwd_params : public Qkv_params {
    // 新增字段
    int *__restrict__ per_row_seqlens_k;  // Per-row seqlen_k values
    // ... 其他字段
};
```

#### flash_api.cpp
- 更新 `set_params_fprop` 和 `set_params_dgrad` 函数
- 添加 `per_row_seqlens_k` 参数传递
- 更新 TORCH_LIBRARY 定义中的函数签名

#### mask.h
```cpp
template <bool Variable_seqlenk_mask=false, ...>
CUTLASS_DEVICE
void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
    // 核心masking逻辑
    if constexpr (Variable_seqlenk_mask) {
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
            int const row_seqlen_k = (row_idx < seqlen_q && per_row_seqlens_k != nullptr) 
                ? per_row_seqlens_k[row_idx] : seqlen_k;
            int const row_seqlenk_col_limit = row_seqlen_k - n_block * kBlockN - thread_col_offset;
            
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= row_seqlenk_col_limit) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
            }
        }
    }
    // ... 其他mask模式
}
```

### Python接口修改

#### flash_attn_interface.py
- 在 `_flash_attn_forward` 中添加维度处理逻辑：
  ```python
  # Handle per_row_seqlens_k: convert from (batch_size, seqlen_q) to flattened
  per_row_seqlens_k_flat = None
  if per_row_seqlens_k is not None:
      if per_row_seqlens_k.dim() == 2:
          per_row_seqlens_k_flat = per_row_seqlens_k.flatten()
      elif per_row_seqlens_k.dim() == 1:
          per_row_seqlens_k_flat = per_row_seqlens_k
  ```
- 更新所有高级接口函数的参数和文档

## 性能考虑

### 计算开销
- **额外内存访问**: 每个query位置需要额外读取一个int32值
- **分支开销**: 每个query位置需要计算不同的列限制
- **缓存友好**: per_row_seqlens_k按行访问，具有良好的空间局部性

### 优化策略
- 使用 `#pragma unroll` 指令优化内层循环
- constexpr 模板特化避免运行时分支判断
- 内存访问模式优化，减少bank conflicts

## 设计约束

### 互斥性
- Variable_seqlenk_mask 不能与 Causal_mask 或 Local_mask 同时使用
- 在编译时通过 static_assert 检查

### 边界检查
- row_idx 必须小于 seqlen_q
- per_row_seqlens_k 指针非空检查
- seqlen_k 值必须在合理范围内

### 数据类型
- per_row_seqlens_k 必须是 int32 类型
- 设备一致性检查

## 错误处理

### 常见错误
1. **维度不匹配**: per_row_seqlens_k 的形状必须是 (batch_size, seqlen_q) 或 (batch_size * seqlen_q,)
2. **设备不匹配**: per_row_seqlens_k 必须与输入张量在同一设备
3. **数据类型错误**: 必须使用 torch.int32
4. **越界访问**: seqlen_k 值超出实际key序列长度

### 调试建议
- 检查 per_row_seqlens_k 的值是否合理
- 确认设备和数据类型一致性
- 验证输出与预期的attention模式匹配

## 测试验证

### 基础测试
- 功能正确性验证
- 梯度计算测试
- 与标准attention的对比
- 边界条件测试

### 性能测试
- 不同序列长度下的性能对比
- 内存使用量分析
- 吞吐量测试

## 总结

Variable_seqlenk_mask为Flash Attention提供了更灵活的masking能力，特别适用于需要动态调整attention范围的应用场景。通过精心的实现和优化，在保持高性能的同时提供了强大的功能扩展。 