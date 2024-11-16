b:
	make build

build:
	zig build
	cp zig-out/bin/zig_matmul .

r:
	make run

run:
	make build
	./zig_matmul

t:
	make test
	
test:
	zig test src/test.zig