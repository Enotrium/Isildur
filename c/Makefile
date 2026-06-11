CC = cc
CFLAGS = -std=c99 -Wall -Wextra -O2 -I.
LDFLAGS = -lm

libisildur.a: isildur.o
	ar rcs $@ $^

isildur.o: isildur.c isildur.h

test_isildur.o: test_isildur.c isildur.h

test: test_isildur.o libisildur.a
	$(CC) $(CFLAGS) -o test_isildur test_isildur.o libisildur.a $(LDFLAGS)
	./test_isildur

demo: demo_isildur.o libisildur.a
	$(CC) $(CFLAGS) -o demo_isildur demo_isildur.o libisildur.a $(LDFLAGS)
	./demo_isildur

demo_isildur.o: demo_isildur.c isildur.h

clean:
	rm -f *.o *.a test_isildur demo_isildur

.PHONY: test demo clean
