## Problem Statement
Suppose we toss darts randomly at a square dartboard, whose bullseye is at the origin, and whose sides are two feet in length. Suppose also that there is a circle inscribed in the square dartboard. The radius of the circle is one foot, and its area is π  square feet. If the points that are hit by the darts are uniformly distributed (and we always hit the square), then the number of darts that hit inside the circle should approximately satisfy the equation:

`Number_in_Circle/Total_Number_of_Tosses = PI/4`

since the ratio of the area of the circle to the area of the square is `PI/4` (π  / 4).

We can use this pseudo code to estimate the value of π  with a random number generator:

```cpp
number_in_circle = 0;
for ( toss = 0; toss < number_of_tosses; toss ++) {
    x = random double between -1 and 1;
    y = random double between -1 and 1;
    distance_squared = x * x + y * y;
    if ( distance_squared <= 1)
        number_in_circle++;
}
pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
```

This is called a [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method), since it uses randomness (the dart tosses). Write a serial C/C++ program that uses the Monte Carlo method to estimate π , with a reasonable number of tosses. You may want to use the type of `long long int` for the number of hits in the circle and the number of tosses, since both may have to be very large to get a reasonable estimate of π.

> Hint: You may want to check if `RAND_MAX` is large enough for use with `rand()` to get a higher precision, and if the number of tosses is great enough to reach a higher accuracy.

## Requirement
Your solution includes two files: a `Makefile` and a `pi.c` (in C) or a `pi.cpp` (in C++).
Your executable program should be named `pi.out` and built with the `make` command.
Your executable program should print out the estimated π  value, which is accurate to two decimal places (i.e., 3.14xx), without any input.
You are supposed to build and run your program by simply typing `make` and `./pi.out`. An example shows as follows.

```bash
$ make
$ ./pi.out
3.1415926....
```

## Performance Profiling
In the future assignments, you will need to parallelize serial versions of applications. In order to identify the “hot code” of a program so as to make an effective improvement, you may need to use profiling tools to find out the program’s bottleneck. Profiling tools, such as `time`, `gprof`, and `perf`, provide you some useful information (e.g., call graphs, execution times, hardware events like cache-miss or branch-miss counts).

### time
The `time` utility executes and times the specified program command with the given arguments. It prints the elapsed time during the execution of a command, time in the system, and execution time of the time command in seconds to standard error.

For example,
```bash
$ time ./pi.out
```

### gprof
`gprof` produces an execution profile of C, Pascal, or Fortran77 programs. The effect of called routines is incorporated in the profile of each caller. The profile data is taken from the call graph profile file (`gmon.out` default) which is created by programs that are compiled with the `-pg` option

Compile the program with the `-pg` option to instrument the program with code that enables profiling.

```bash
$ gcc -pg pi.c -o pi.out
```

Now, just execute the executable file to generate `gmon.out`.

```bash
$ ./pi.out
```

After `gmon.out` is generated, the gprof tool is run with the executable name and the above generated `gmon.out` as arguments. The `-b` option suppresses the printing of a description of each field in the profile. You may also redirect stdout to a file (say, `profiling_result`) for later use.

```bash
$ gprof ./pi.out gmon.out -b > profiling_result
```

Read the `profiling_result` file to get the profiling information.

> Note: If the execution time is less than 0.01 seconds, no time will be counted.

### perf
The `perf` command is used as a primary interface to the Linux kernel performance monitoring capabilities and can record CPU performance counters and trace points. Similar to `gprof`, before you profile the program, you need to compile the program with `-g` option to enable profiling.

```bash
$ gcc -g pi.c -o pi.out
```

Use the `perf list` command to list available events.

Let’s say if we are interested in the `cpu-cycles` event, then we just run the program using the `perf` tool with the event(s) we want to monitor.

```bash
$ perf record -e cpu-cycles  ./pi.out
```

Now you can read the profiling result by running

```bash
$ perf report
```

## Reference
- [Official document of GNU MAKE](https://www.gnu.org/software/make/manual/make.html)
- [perf](https://www.brendangregg.com/perf.html)
- [Performance Profiling](https://www.clear.rice.edu/comp321/html/laboratories/lab07/)
