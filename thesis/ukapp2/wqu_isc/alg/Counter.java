package counter;

import java.util.Random;

/**
 * A simple, potentially shared concurrent, counter class which can print a trace of its behaviour.
 * The non-static methods publicly available are
 * <ul>
 * <li> void startCount(): start the counter
 * <li> void stepCount(): step the counter
 * <li> boolean isFinished(): has the counter passed its limit?
 * </ul>
 * The following static methods are also available which will modify the behaviour of all
 * Counters
 * <ul>
 * <li> void traceOn(): print a trace of changes to counters to stdout
 * <li> void traceOff(): do not print a trace
 * <li> void setDelay(int delay): slow down counters to increase the chance of time slicing
 * </ul>
 * 
 * @author Hugh Osborne
 * @version November 2019
 **/

public class Counter extends Thread
{
    /**
     * Counting starts at the value "from", takes place in increments of "step", and 
     * terminates when the counter passes the "limit" value.
     **/
    private int from,limit,step;
    
    /**
     * The counter is shared between all instances of Counter.
     **/
    private static int counter;

    /**
     * The most detailed constructor, allowing for specification of all parameters of the counter.
     *
     * @param name    the name of this counter
     * @param from the value at which the counter starts
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     * @param step    the amount the counter will be incremented by
     * 
     * @throws CounterException if:
     * <ul>
     * <li> The increment is set to 0 ("step" == 0)
     * <li> The counter counts in the "wrong direction" (e.g. from 0 to 10 in steps of -1)
     *   <ul>
     *   <li> If "limit" &gt; "from" then "step" must be positive
     *   <li> If "limit" &lt; "from" then "step" must be negative
     *   <li> If "limit" == "from" then it doesn't matter
     *   </ul>
     * I.e. "step"*("limit"-"from") must always be &ge; 0
     * </ul>
     **/
    private Counter(String name, int from, int limit, int step) throws CounterException {
        super(name);
        this.from = from;
        this.limit = limit;
        if (step == 0) { // Increment is zero
            throw new CounterException("Counter's increment must be non-zero");
        } else if (step * (limit-from) < 0) { // Increment is in "the wrong direction"
            throw new CounterException("Counter does not count in the right direction to reach its limit");
        } else {
            this.step = step;
        }
    }
    
    /**
     * Constructor for a "nameless" counter.  All other parameters can be specified.
     * A name is generated for the Counter summarising the values of its parameters.
     *
     * @param from the value at which the counter starts
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     * @param step    the amount the counter will be incremented by
     * @throws CounterException if:
     * <ul>
     * <li> The increment is set to 0 ("step" == 0)
     * <li> The counter counts in the "wrong direction" (e.g. from 0 to 10 in steps of -1)
     *   <ul>
     *   <li> If "limit" &gt; "from" then "step" must be positive
     *   <li> If "limit" &lt; "from" then "step" must be negative
     *   <li> If "limit" == "from" then it doesn't matter
     *   </ul>
     * I.e. "step"*("limit"-"from") must always be &ge; 0
     * </ul>
     **/
    Counter(int from,int limit,int step) throws CounterException {
        this("Counter(" + from + "=[" + (step >=0 ? "+" : "") + step + "]=>" + limit + ")",from,limit,step);
    }
    
    /**
     * Constructor where the step size is not specified.
     * The step size is set to count by ones in the right direction. I.e.
     * <ul>
     * <li> If "limit" &gt; "from" then "step" is set to +1
     * <li> If "limit" &lt; "from" then "step" is set to -1
     * <li> If "limit" == "from" then it doesn't matter what "step" is (here it is set to -1)
     * </ul>
     *
     * @param name    the name of this counter
     * @param from the value at which the counter starts
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     **/
    Counter(String name,int from,int limit) {
        super(name);
        this.from = from;
        this.limit = limit;
        if (limit > from) {
            step = 1;
        } else {
            step = -1;
        }
    }
    
    /**
     * "Nameless", "stepless" constructor.
     * A name is generated for the Counter summarising the values of its parameters.
     * The step size is set to count by ones in the right direction. I.e.
     * <ul>
     * <li> If "limit" &gt; "from" then "step" is set to +1
     * <li> If "limit" &lt; "from" then "step" is set to -1
     * <li> If "limit" == "from" then it doesn't matter what "step" is (here it is set to -1)
     * </ul>
     *
     * @param from the value at which the counter starts
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     **/
    Counter(int from,int limit) {
        this("Counter(" + from + "=[" + (limit > from ? "+1" : "-1") + "]=>" + limit + ")",from,limit);
    }

    /**
     * Constructor where neither the step size, nor the initial count is specified.
     * Counting will start at zero.
     * The step size is set to count by ones in the right direction. I.e.
     * <ul>
     * <li> If "limit" &gt; 0 then "step" is set to +1
     * <li> If "limit" &lt; 0 then "step" is set to -1
     * <li> If "limit" == 0 then it doesn't matter what "step" is (here it is set to -1)
     * </ul>
     *
     * @param name    the name of this counter
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     **/
    Counter(String name,int limit) {
        this(name,0,limit);
    }

    /**
     * A "nameless", "stepless" constructor with no initial count.
     * @param limit   the limiting value (if the counter passes this value counting will stop)
     * A name is generated for the Counter summarising the values of its parameters.
     * Counting will start at zero.
     * The step size is set to count by ones in the right direction. I.e.
     * <ul>
     * <li> If "limit" &gt; 0 then "step" is set to +1
     * <li> If "limit" &lt; then "step" is set to -1
     * <li> If "limit" == 0 then it doesn't matter what "step" is (here it is set to -1)
     * </ul>
     **/
    Counter(int limit) {
        this(0,limit);
    }

    /**
     * Provide a facility for switching tracing on/off.  All counters share the same tracing
     * toggle.
     **/
    private static boolean tracingOn = false;
    
    /**
     * Switch tracing on.
     **/
    public static void traceOn() {
        tracingOn = true;
    }
    
    /**
     * Switch tracing off.
     **/
    public static void traceOff() {
        tracingOn = false;
    }
    
    /**
     * A counter will pause, or delay, between each action (starting the count, checking the count's value, stepping
     * the count) by a random time between a minimum and a maximum delay.  Delay times are specified in milliseconds.
     **/
    private static int minimumDelay = 0, maximumDelay = 10;
    
    /**
     * Set the maximum delay in seconds.
     * @param maximum the maximum delay required, in seconds.
     * The minimum delay is set to zero.
     * @throws CounterException if the requested delay is less than zero seconds.
     **/
    protected static void setDelay(double maximum) throws CounterException {
        if (maximum < 0) {
            throw new CounterException("Attempt to set the delay interval to 0" + "==>" + maximum + ". The maximum delay can not be less than the minimum");
        }
        minimumDelay = 0;
        maximumDelay = (int) (maximum*1000);
    }
    
    /**
     * Set both the minimum and maximum delay in seconds.
     * @param minimum the minimum delay desired, in seconds.
     * @param maximum the maximum delay desired, in seconds.
     * @throws CounterException if the requested maximum delay is not greater than or equal to the requested minimum, or
     *         if either of the specified delays is negative.
     */
    protected static void setDelay(double minimum,double maximum) throws CounterException {
        if (minimum < 0 || maximum < minimum) {
            throw new CounterException("Attempt to set the delay interval to " + minimum + "==>" + maximum + ". The maximum delay can not be less than the minimum");
        }
        minimumDelay = (int) (minimum*1000);
        maximumDelay = (int) (maximum*1000);
    }

    /**
     * Used to generate random delays between the minimum and maximum delay values.
     */
    private static Random random = new Random();
    
    /**
     * Internal delay method for slowing down the counter
     **/
    private void delay() {
        try {
            sleep(minimumDelay + random.nextInt(maximumDelay - minimumDelay));
        } catch (InterruptedException ie) {}
    }
    
    /**
     * Start the counter by setting it to the initial value
     **/
    public void startCount() {
    	delay();
        counter = from;
        if (tracingOn) System.out.println(getName() + " has started: " + counter);
    }
    
    /**
     * Increment the counter.
     **/
    public void stepCount() {
        delay();
        counter += step;
        if (tracingOn) System.out.println(getName() + " has stepped: " + counter);
    }

    /**
     * Check whether the counter has passed its limiting value.  If the increment is positive
     * the counter has passed its limit if it is greater than the limit.  If the increment is
     * less than zero the counter must be lower than its limit.
     * @return true iff this counter has passed its limiting value.
     **/
    public boolean isFinished() {
    	delay();
        boolean finished = 
          (step > 0 && counter >= limit) || (step < 0 && counter <= limit);
        if (tracingOn && finished) System.out.println(getName() + " has finished: " + counter);
        return finished;
    }

    /**
     * Run this counter.
     */
    public void run() {
        /*
         * A straightforward implementation.  Use startCount() to start the counter off, then step the counter, using
         * stepCount(), in a loop, uing isFinished() to test when the loop should terminate.
         */
        startCount();
        while (!isFinished()) {
            stepCount();
        }
    }
}
