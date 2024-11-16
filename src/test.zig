const std = @import("std");

const tensor = @import("tensor.zig");

test "test add" {
    const allocator = std.heap.page_allocator;

    var a = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer a.deinit();

    const b = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer b.deinit();

    a = try a.add(&b);

    try std.testing.expect(std.mem.eql(f16, a.data, &.{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 }));
}

test "test sub" {
    const allocator = std.heap.page_allocator;

    var a = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer a.deinit();

    var b = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer b.deinit();

    b = try b.div_scalar(@floatFromInt(b.numel()));

    a = try a.sub(&b);

    try std.testing.expect(std.mem.eql(f16, a.data, &.{ 0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1 }));
}

test "test mul and div" {
    const allocator = std.heap.page_allocator;

    var a = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer a.deinit();

    var b = try tensor.Tensor(f16).arange(0, 10, allocator);
    defer b.deinit();

    b = try b.div(&try b.add_scalar(@floatFromInt(1)));

    a = try a.mul(&b);

    try std.testing.expect(std.mem.eql(f16, a.data, &.{ 0, 0.5, 1.333, 2.25, 3.2, 4.168, 5.14, 6.125, 7.11, 8.1 }));
}
