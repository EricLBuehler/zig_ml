const std = @import("std");
const mem = std.mem;

pub const Error = error{
    OutOfMemory,
    NumelMismatch,
    PermuteIndicesLengthMismatch,
    TransposeRequiresMin2Dims,
    ShapeMismatch,
    DTypeMismatch,
    MatmulBatchMismatch,
    MatmulInnerMismatch,
    InvalidDataLength,
    InvalidDim,
};

fn contiguous_stride(shape: []const usize, allocator: mem.Allocator) Error![]usize {
    const stride = try allocator.alloc(usize, shape.len);
    @memset(stride, 1);

    var prod: usize = 1;
    for (0..shape.len) |i| {
        const idx = shape.len - i - 1;
        const u = shape[idx];
        stride[idx] = prod;
        prod *= u;
    }

    return stride;
}

fn recursive_write(comptime T: type, buffer: []T, writer: std.ArrayList(u8).Writer, indices: []const usize, shape: []const usize, strides: []const usize) Error!void {
    if (shape.len == indices.len) {
        var offset: usize = 0;
        for (indices, strides) |idx, stride| {
            offset += idx * stride;
        }
        _ = try writer.print("{d}", .{buffer[offset]});
    } else {
        _ = try writer.print("[", .{});
        for (0..shape[indices.len]) |i| {
            if (i > 0) {
                _ = try writer.print(",", .{});
            }

            var allocator = std.heap.page_allocator;
            const new_indices = try allocator.alloc(usize, indices.len + 1);
            defer allocator.free(new_indices);

            var idx: usize = 0;
            while (idx < indices.len) {
                new_indices[idx] = indices[idx];
                idx += 1;
            }
            new_indices[idx] = i;
            try recursive_write(T, buffer, writer, new_indices, shape, strides);
        }
        _ = try writer.print("]", .{});
    }
    return;
}

fn numel_shape(shape: []const usize) usize {
    var accum: usize = 1;

    for (shape) |i| {
        accum *= i;
    }
    return accum;
}

pub fn Tensor(
    comptime T: type,
) type {
    return struct {
        is_view: bool,
        stride: []const usize,
        shape: []const usize,
        data: []T,
        allocator: mem.Allocator,

        /// Create a new tensor based on data with assumed contiguous strides.
        ///
        /// ## Errors
        /// - `NumelMismatch`: shape does not match the data.
        /// - `InvalidDataLength`: shape has 0 elements.
        ///
        /// ## Semantics
        /// - `shape` is not consumed and a copy is made.
        /// - `data` is consumed.
        pub fn from_slice(
            shape: []const usize,
            data: []T,
            allocator: mem.Allocator,
        ) Error!Tensor(T) {
            const shape_cpy = try allocator.alloc(usize, shape.len);
            @memcpy(shape_cpy, shape);
            if (data.len != numel_shape(shape)) {
                return error.NumelMismatch;
            }
            if (data.len == 0) {
                return error.InvalidDataLength;
            }
            const stride = try contiguous_stride(shape, allocator);
            return Tensor(T){ .stride = stride, .shape = shape_cpy, .data = data, .allocator = allocator, .is_view = false };
        }

        /// Create a new tensor from start to end.
        ///
        /// ## Errors
        /// - `InvalidDataLength`: `end <= start`
        ///
        /// ## Semantics
        /// - `shape` is not consumed and a copy is made.
        /// - `data` is consumed.
        pub fn arange(
            start: T,
            end: T,
            allocator: mem.Allocator,
        ) Error!Tensor(T) {
            if (end <= start) {
                return error.InvalidDataLength;
            }
            const len = if (@typeInfo(T) == .Float) @as(usize, @intFromFloat(end - start)) else end - start;
            const data = try allocator.alloc(T, len);
            for (0..len, 0..) |v, i| {
                data[i] = if (@typeInfo(T) == .Float) @as(T, @floatFromInt(v)) + start else @as(T, @intCast(v)) + start;
            }

            const shape = try allocator.alloc(usize, 1);
            shape[0] = len;
            const stride = try contiguous_stride(shape, allocator);
            return Tensor(T){ .stride = stride, .shape = shape, .data = data, .allocator = allocator, .is_view = false };
        }

        /// Create a new tensor initialized with values randomly distributed between [0, 1).
        ///
        /// ## Errors
        /// - `InvalidDataLength`: shape has 0 elements
        ///
        /// ## Semantics
        /// - `shape` is not consumed and a copy is made.
        pub fn rand(
            shape: []const usize,
            seed: u64,
            allocator: mem.Allocator,
        ) Error!Tensor(T) {
            const data = try allocator.alloc(T, numel_shape(shape));
            for (data) |*elem| {
                var rng = std.rand.Isaac64.init(seed);
                elem.* = rng.random().float(T);
            }

            const shape_cpy = try allocator.alloc(usize, shape.len);
            @memcpy(shape_cpy, shape);
            if (data.len != numel_shape(shape)) {
                return error.NumelMismatch;
            }
            if (data.len == 0) {
                return error.InvalidDataLength;
            }

            const stride = try contiguous_stride(shape, allocator);
            return Tensor(T){ .stride = stride, .shape = shape_cpy, .data = data, .allocator = allocator, .is_view = false };
        }

        /// Create a new tensor initialized with values normally distributed with a mean of 0.0 and a std deviation of 1.0.
        ///
        /// ## Errors
        /// - `InvalidDataLength`: shape has 0 elements
        ///
        /// ## Semantics
        /// - `shape` is not consumed and a copy is made.
        pub fn randn(
            shape: []const usize,
            seed: u64,
            allocator: mem.Allocator,
        ) Error!Tensor(T) {
            const data = try allocator.alloc(T, numel_shape(shape));
            for (data) |*elem| {
                var rng = std.rand.Isaac64.init(seed);
                elem.* = rng.random().floatNorm(T);
            }

            const shape_cpy = try allocator.alloc(usize, shape.len);
            @memcpy(shape_cpy, shape);
            if (data.len == 0) {
                return error.InvalidDataLength;
            }

            const stride = try contiguous_stride(shape, allocator);
            return Tensor(T){ .stride = stride, .shape = shape_cpy, .data = data, .allocator = allocator, .is_view = false };
        }

        pub fn deinit(self: *const Tensor(T)) void {
            if (!self.is_view) {
                self.allocator.free(self.data);
            }
            self.allocator.free(self.stride);
            self.allocator.free(self.shape);
        }

        // ================================================================================
        // SHAPE MANIPULATION
        // ================================================================================

        /// Number of elements
        pub fn numel(self: *const Tensor(T)) usize {
            return numel_shape(self.shape);
        }

        /// Returns `true` if the tensor is row-contiguous.
        pub fn is_contiguous(self: *const Tensor(T)) bool {
            var acc: usize = 1;
            for (0..self.shape.len) |i| {
                const idx = self.shape.len - i - 1;
                const dim = self.shape[idx];
                const stride = self.stride[idx];
                if (dim > 1 and stride != acc) {
                    return false;
                }
                acc *= dim;
            }
            return true;
        }

        /// Reshape this tensor. A view is returned. At most one dimension may be -1, this infers the dimension.
        ///
        /// ## Errors
        /// - `NumelMismatch`: number of elements do not match.
        /// - `InvalidDim`: multiple -1 or 0s encountered in new shape.
        ///
        /// ## Semantics
        /// - `shape` is not consumed and a copy is made.
        pub fn view(self: *const Tensor(T), shape: []const i64) Error!Tensor(T) {
            var num_infer: usize = 0;
            var infer_dim: usize = 0;
            for (shape, 0..) |dim, i| {
                if (dim == -1) {
                    num_infer += 1;
                    infer_dim = i;
                } else if (dim == 0) {
                    return error.InvalidDim;
                }
            }
            if (num_infer > 1) {
                return error.InvalidDim;
            }

            const shape_cpy = try self.allocator.alloc(usize, shape.len);

            if (num_infer > 0) {
                var other_dims: i64 = 1;
                for (shape, 0..) |dim, i| {
                    if (i != infer_dim) {
                        other_dims *= dim;
                    }
                }
                const inferred_dim = self.numel() / @as(usize, @intCast(other_dims));
                shape_cpy[infer_dim] = inferred_dim;
            }
            for (shape, 0..) |new_dim, i| {
                if (i != infer_dim or num_infer == 0) {
                    shape_cpy[i] = @as(usize, @intCast(new_dim));
                }
            }
            if (self.numel() != numel_shape(shape_cpy)) {
                return error.NumelMismatch;
            }
            const stride = try contiguous_stride(shape_cpy, self.allocator);
            return Tensor(T){ .stride = stride, .shape = shape_cpy, .data = self.data, .allocator = self.allocator, .is_view = true };
        }

        /// Reorder the dimensions of this tensor.
        ///
        /// ## Semantics
        /// - `indices` is not consumed.
        pub fn permute(self: *const Tensor(T), indices: []const usize) Error!Tensor(T) {
            if (self.shape.len != indices.len) {
                return error.PermuteIndicesLengthMismatch;
            }

            var shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            var stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            for (indices, 0..) |idx, i| {
                stride_cpy[i] = self.stride[idx];
                shape_cpy[i] = self.shape[idx];
            }
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = self.data, .allocator = self.allocator, .is_view = true };
        }

        /// Transpose the last two dimensions of the tensor
        pub fn t(self: *const Tensor(T)) Error!Tensor(T) {
            if (self.shape.len < 2) {
                return error.TransposeRequiresMin2Dims;
            }

            var indices = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(indices);

            for (indices, 0..) |*idx, i| {
                idx.* = i;
            }
            indices[indices.len - 2] = indices.len - 1;
            indices[indices.len - 1] = indices.len - 2;
            return try self.permute(indices);
        }

        // ================================================================================
        // BINARY OPERATORS (Tensor - Tensor)
        // ================================================================================

        /// Elementwise addition of two tensors.
        ///
        /// `out[i] = self[i] + other[i]`
        pub fn add(self: *const Tensor(T), other: *const Tensor(T)) Error!Tensor(T) {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, other.data, 0..) |a, b, i| {
                arr[i] = a + b;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise subtraction of two tensors.
        ///
        /// `out[i] = self[i] - other[i]`
        pub fn sub(self: *const Tensor(T), other: *const Tensor(T)) Error!Tensor(T) {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, other.data, 0..) |a, b, i| {
                arr[i] = a - b;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise multiplication of two tensors.
        ///
        /// `out[i] = self[i] * other[i]`
        pub fn mul(self: *const Tensor(T), other: *const Tensor(T)) Error!Tensor(T) {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, other.data, 0..) |a, b, i| {
                arr[i] = a * b;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise division of two tensors.
        ///
        /// `out[i] = self[i] / other[i]`
        pub fn div(self: *const Tensor(T), other: *const Tensor(T)) Error!Tensor(T) {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return error.ShapeMismatch;
            }
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, other.data, 0..) |a, b, i| {
                arr[i] = a / b;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Compute matrix multiplication of `self` and `rhs`.
        ///
        /// - `self`: `(b_1, b_2, ..., b_n, i, k)`
        /// - `rhs`: `(b_1, b_2, ..., b_n, k, j)`
        /// - Output: `(b_1, b_2, ..., b_n, i, j)`
        ///
        /// ## Errors
        /// - `MatmulBatchMismatch`: if the batch dimensions do not match
        /// - `MatmulInnerMismatch`: if `k` does not match
        pub fn matmul(self: *const Tensor(T), rhs: *const Tensor(T)) Error!Tensor(T) {
            const self_batch: usize = self.numel() / (self.shape[self.shape.len - 1] * self.shape[self.shape.len - 2]);
            const self_stride = try contiguous_stride(&.{ self_batch, self.shape[self.shape.len - 1], self.shape[self.shape.len - 2] }, self.allocator);
            defer self.allocator.free(self_stride);

            const rhs_batch: usize = rhs.numel() / (rhs.shape[rhs.shape.len - 1] * rhs.shape[rhs.shape.len - 2]);
            const rhs_stride = try contiguous_stride(&.{ rhs_batch, rhs.shape[rhs.shape.len - 1], rhs.shape[rhs.shape.len - 2] }, self.allocator);
            defer self.allocator.free(rhs_stride);

            if (self_batch != rhs_batch) {
                return error.MatmulBatchMismatch;
            }

            const i_dim = self.shape[self.shape.len - 2];
            const j_dim = rhs.shape[self.shape.len - 1];
            const k_dim = rhs.shape[rhs.shape.len - 2];
            const b_dim = self_batch;

            if (rhs.shape[rhs.shape.len - 2] != self.shape[self.shape.len - 1]) {
                return error.MatmulInnerMismatch;
            }

            const arr = try self.allocator.alloc(T, self.data.len);
            @memset(arr, 0);

            const out_stride = try contiguous_stride(&.{ self_batch, i_dim, j_dim }, self.allocator);

            for (0..b_dim) |b| {
                for (0..i_dim) |i| {
                    for (0..j_dim) |j| {
                        for (0..k_dim) |k| {
                            const out_idx = b * out_stride[out_stride.len - 3] + i * out_stride[out_stride.len - 2] + j * out_stride[out_stride.len - 1];
                            const self_idx = b * self_stride[self_stride.len - 3] + i * self_stride[self_stride.len - 2] + k * self_stride[self_stride.len - 1];
                            const rhs_idx = b * rhs_stride[rhs_stride.len - 3] + k * rhs_stride[rhs_stride.len - 1] + j * rhs_stride[rhs_stride.len - 2];
                            arr[out_idx] += self.data[self_idx] * rhs.data[rhs_idx];
                        }
                    }
                }
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy[0 .. self.shape.len - 2], self.shape[0 .. self.shape.len - 2]);
            shape_cpy[shape_cpy.len - 2] = i_dim;
            shape_cpy[shape_cpy.len - 1] = j_dim;
            const new_stride = try contiguous_stride(shape_cpy, self.allocator);

            return Tensor(T){ .stride = new_stride, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Compute matrix multiplication of `self` and `rhs`.
        /// `rhs` is transposed from the `matmul` version.
        ///
        /// - `self`: `(b_1, b_2, ..., b_n, i, k)`
        /// - `rhs`: `(b_1, b_2, ..., b_n, j, k)`
        /// - Output: `(b_1, b_2, ..., b_n, i, j)`
        ///
        /// ## Errors
        /// - `MatmulBatchMismatch`: if the batch dimensions do not match
        /// - `MatmulInnerMismatch`: if `k` does not match
        pub fn matmul_rhst(self: *const Tensor(T), rhs: *const Tensor(T)) Error!Tensor(T) {
            const self_batch: usize = self.numel() / (self.shape[self.shape.len - 1] * self.shape[self.shape.len - 2]);
            const self_stride = try contiguous_stride(&.{ self_batch, self.shape[self.shape.len - 1], self.shape[self.shape.len - 2] }, self.allocator);
            defer self.allocator.free(self_stride);

            const rhs_batch: usize = rhs.numel() / (rhs.shape[rhs.shape.len - 1] * rhs.shape[rhs.shape.len - 2]);
            const rhs_stride = try contiguous_stride(&.{ rhs_batch, rhs.shape[rhs.shape.len - 1], rhs.shape[rhs.shape.len - 2] }, self.allocator);
            defer self.allocator.free(rhs_stride);

            if (self_batch != rhs_batch) {
                return error.MatmulBatchMismatch;
            }

            const i_dim = self.shape[self.shape.len - 2];
            const j_dim = rhs.shape[self.shape.len - 1];
            const k_dim = rhs.shape[rhs.shape.len - 2];
            const b_dim = self_batch;

            if (rhs.shape[rhs.shape.len - 2] != self.shape[self.shape.len - 1]) {
                return error.MatmulInnerMismatch;
            }

            const arr = try self.allocator.alloc(T, self.data.len);
            @memset(arr, 0);

            const out_stride = try contiguous_stride(&.{ self_batch, i_dim, j_dim }, self.allocator);

            for (0..b_dim) |b| {
                for (0..i_dim) |i| {
                    for (0..j_dim) |j| {
                        for (0..k_dim) |k| {
                            const out_idx = b * out_stride[out_stride.len - 3] + i * out_stride[out_stride.len - 2] + j * out_stride[out_stride.len - 1];
                            const self_idx = b * self_stride[self_stride.len - 3] + i * self_stride[self_stride.len - 2] + k * self_stride[self_stride.len - 1];
                            const rhs_idx = b * rhs_stride[rhs_stride.len - 3] + j * rhs_stride[rhs_stride.len - 2] + k * rhs_stride[rhs_stride.len - 1];
                            arr[out_idx] += self.data[self_idx] * rhs.data[rhs_idx];
                        }
                    }
                }
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy[0 .. self.shape.len - 2], self.shape[0 .. self.shape.len - 2]);
            shape_cpy[shape_cpy.len - 2] = i_dim;
            shape_cpy[shape_cpy.len - 1] = j_dim;
            const new_stride = try contiguous_stride(shape_cpy, self.allocator);

            return Tensor(T){ .stride = new_stride, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        // ================================================================================
        // BINARY OPERATORS (Tensor - scalar)
        // ================================================================================

        /// Elementwise addition of a tensor and a scalar.
        ///
        /// `out[i] = self[i] + scalar`
        pub fn add_scalar(self: *const Tensor(T), scalar: T) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = a + scalar;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise subtraction of a tensor and a scalar.
        ///
        /// `out[i] = self[i] - scalar`
        pub fn sub_scalar(self: *const Tensor(T), scalar: T) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = a - scalar;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise multiplication of a tensor and a scalar.
        ///
        /// `out[i] = self[i] * scalar`
        pub fn mul_scalar(self: *const Tensor(T), scalar: T) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = a * scalar;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise division of a tensor and a scalar.
        ///
        /// `out[i] = self[i] . scalar`
        pub fn div_scalar(self: *const Tensor(T), scalar: T) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = a / scalar;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        // ================================================================================
        // UNARY OPERATORS (Tensor)
        // ================================================================================

        /// Elementwise negation of a tensor.
        ///
        /// `out[i] = -self[i]`
        pub fn neg(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = -a;
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise squaring of a tensor.
        ///
        /// `out[i] = std.math.pow(self[i], 2)`
        pub fn sqr(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.pow(a, 2);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise square root of a tensor.
        ///
        /// `out[i] = std.math.sqrt(self[i])`
        pub fn sqrt(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.sqrt(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise base-e exponential of a tensor.
        ///
        /// `out[i] = std.math.exp(self[i])`
        pub fn exp(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.exp(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise cosine of a tensor.
        ///
        /// `out[i] = std.math.cos(self[i])`
        pub fn cos(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.cos(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise sin of a tensor.
        ///
        /// `out[i] = std.math.sin(self[i])`
        pub fn sin(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.sin(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise tangent of a tensor.
        ///
        /// `out[i] = std.math.tan(self[i])`
        pub fn tan(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.tan(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise hyperbolic cosine of a tensor.
        ///
        /// `out[i] = std.math.cosh(self[i])`
        pub fn cosh(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.cosh(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise hyperbolic sin of a tensor.
        ///
        /// `out[i] = std.math.sinh(self[i])`
        pub fn sinh(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.sinh(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        /// Elementwise hyperbolic tangent of a tensor.
        ///
        /// `out[i] = std.math.tanh(self[i])`
        pub fn tanh(self: *const Tensor(T)) Error!Tensor(T) {
            const arr = try self.allocator.alloc(T, self.data.len);
            for (self.data, 0..) |a, i| {
                arr[i] = std.math.tanh(a);
            }

            const shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            const stride_cpy = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_cpy, self.stride);
            return Tensor(T){ .stride = stride_cpy, .shape = shape_cpy, .data = arr, .allocator = self.allocator, .is_view = false };
        }

        // ================================================================================
        // REDUCTION OPERATORS (Tensor)
        // ================================================================================

        /// Sum the tensor in certain dimensions. Each dimension is replaced by a 1.
        ///
        /// Note: The dimensions must be in increasing order.
        ///
        /// ## Semantics
        /// - `dims` are not consumed.
        pub fn sum(self: *const Tensor(T), dims: []const usize) Error!Tensor(T) {
            var shape_cpy = try self.allocator.alloc(usize, self.shape.len);
            @memcpy(shape_cpy, self.shape);
            for (dims) |dim| {
                shape_cpy[dim] = 1;
            }
            const new_stride = try contiguous_stride(shape_cpy, self.allocator);

            const dst = try self.allocator.alloc(T, numel_shape(shape_cpy));
            for (self.data, 0..) |src, unstr_idx| {
                var dst_index = unstr_idx;
                for (dims) |dim| {
                    const pre = dst_index / self.stride[dim];
                    const post = dst_index % self.stride[dim];
                    dst_index = (pre / self.shape[dim]) * self.stride[dim] + post;
                }
                dst[dst_index] += src;
            }
            return Tensor(T){ .stride = new_stride, .shape = shape_cpy, .data = dst, .allocator = self.allocator, .is_view = false };
        }

        /// Mean of the tensor in certain dimensions. Each dimension is replaced by a 1.
        ///
        /// Note: The dimensions must be in increasing order.
        ///
        /// ## Semantics
        /// - `dims` are not consumed.
        pub fn mean(self: *const Tensor(T), dims: []const usize) Error!Tensor(T) {
            const summed = try self.sum(dims);
            var numel_dims: usize = 1;
            for (dims) |dim| {
                numel_dims *= self.shape[dim];
            }
            const scale = if (@typeInfo(T) == .Float) @as(T, @floatFromInt(numel_dims)) else @as(T, @intCast(numel_dims));
            return summed.div_scalar(scale);
        }

        /// Sum of the tensor in all certain dimensions. This returns a tensor of shape [1].
        ///
        /// ## Semantics
        /// - `dims` are not consumed.
        pub fn sum_all(self: *const Tensor(T)) Error!Tensor(T) {
            const dims = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(dims);
            for (0..self.shape.len) |i| {
                dims[i] = i;
            }
            return self.sum(dims);
        }

        /// Mean of the tensor in all certain dimensions. This returns a tensor of shape [1].
        ///
        /// ## Semantics
        /// - `dims` are not consumed.
        pub fn mean_all(self: *const Tensor(T)) Error!Tensor(T) {
            const summed = try self.sum_all();
            const scale = if (@typeInfo(T) == .Float) @as(T, @floatFromInt(self.numel())) else @as(T, @intCast(self.numel()));
            return summed.div_scalar(scale);
        }

        // ================================================================================
        // UTILITIES
        // ================================================================================

        /// Return a string debug representation.
        ///
        /// ## Semantics
        /// - The caller is responsible for deallocating the memory.
        pub fn debug(self: *const Tensor(T)) Error!std.ArrayList(u8) {
            var arrlist = std.ArrayList(u8).init(self.allocator);
            var writer = arrlist.writer();
            _ = try writer.write("Tensor(");
            try recursive_write(T, self.data, writer, &.{}, self.shape, self.stride);
            _ = try writer.write(")\n");
            return arrlist;
        }

        /// Print a debug representation.
        pub fn print(self: *const Tensor(T)) Error!void {
            const repr = try self.debug();
            defer repr.deinit();
            std.debug.print("{s}", .{repr.items});
            return;
        }
    };
}
