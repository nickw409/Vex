package log

import (
	"fmt"
	"os"
	"time"
)

var start = time.Now()

func elapsed() string {
	d := time.Since(start)
	return fmt.Sprintf("%6.1fs", d.Seconds())
}

// Info logs a timestamped message to stderr.
func Info(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(os.Stderr, "[vex %s] %s\n", elapsed(), msg)
}
