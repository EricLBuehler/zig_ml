b:
	make build

build:
	zig build
	cp zig-out/bin/zig_ml .

r:
	make run

run:
	make build
	./zig_ml

t:
	make test
	
test:
	zig test src/test.zig