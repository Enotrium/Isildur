CC = cc
CFLAGS = -std=c99 -Wall -Wextra -O2 -I.

libisildur.a: isildur.o
	ar rcs $@ $^

isildur.o: isildur.c isildur.h

test: test_isildur.c isildur.h libisildur.a
	$(CC) $(CFLAGS) -o test_isildur test_isildur.c libisildur.a
	./test_isildur

clean:
	rm -f *.o *.a test_isildur

.PHONY: test clean
