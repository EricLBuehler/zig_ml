const std = @import("std");

const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");

// Return a random number from 0 to given `max`
fn make_randn(seed: u64) f32 {
    var rng = std.rand.Isaac64.init(seed);
    return rng.random().float(f32);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const bytes = try safetensors.read_bytes("model-00001-of-00002.safetensors", allocator);
    defer allocator.free(bytes);
    const header_slice = safetensors.get_header_slice(bytes);
    const header: []u8 = header_slice[0];
    const data_offset: u64 = header_slice[1];

    const parsed_header = try safetensors.parse_header(header, allocator);
    defer parsed_header.deinit();
    for (parsed_header.items) |item| {
        defer item.deinit();
        const tensor_data = bytes[item.data_offsets[0] + data_offset .. item.data_offsets[1] + data_offset];
        if (!std.mem.eql(u8, item.dtype, "F32")) {
            std.debug.print("ERROR: unsupported dtype `{s}`.\n", .{item.dtype});
            std.debug.assert(false);
        }
        const ty = f32;
        const float_data = std.mem.bytesAsSlice(ty, @as([]align(@alignOf(ty)) u8, @alignCast(tensor_data)));
        const new_data = try allocator.alloc(ty, float_data.len);
        @memcpy(new_data, float_data);
        const x = try tensor.Tensor(ty).from_slice(item.shape, new_data, allocator);
        defer x.deinit();

        const mean = try x.mean_all();
        const repr = try mean.debug();
        defer repr.deinit();
        std.debug.print("{s}: {any}, mean = {s}\n", .{ item.name, x.shape, repr.items });
    }

    std.debug.print("\n\n\n", .{});

    const N: usize = 2;

    var a = try tensor.Tensor(f32).arange(0, 3 * N * N, allocator);
    defer a.deinit();

    a = try a.div_scalar(@floatFromInt(a.numel()));
    a = try a.view(&.{ 3, N, N });

    a = try a.mean(&.{ 0, 1, 2 });
    a = try a.view(&.{-1});

    try a.print();

    // var b = try tensor.Tensor(f32).arange(0, N * N, allocator);
    // defer b.deinit();

    // b = try b.div_scalar(@floatFromInt(b.numel()));
    // b = try b.view(&.{ N, N });
    // b = try b.t();

    // const start1 = try std.time.Instant.now();
    // const c1 = try a.matmul(&b);
    // const ns_duration1 = @as(f32, @floatFromInt((try std.time.Instant.now()).since(start1))) / @as(f32, @floatFromInt(std.time.ns_per_ms));
    // std.debug.print("matmul took {}ms\n", .{ns_duration1});

    // std.debug.print("a: ", .{});
    // std.debug.print("{any}\n", .{a.shape});
    // // try a.print();
    // std.debug.print("b: ", .{});
    // std.debug.print("{any}\n", .{b.shape});
    // // try b.print();
    // std.debug.print("c1: ", .{});
    // std.debug.print("{any}\n", .{c1.shape});
    // // try c1.print();
}
