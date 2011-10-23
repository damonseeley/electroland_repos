package net.electroland.eio.filters;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class SimpleLowPass implements IOFilter {

    private static Logger logger = Logger.getLogger(IOFilter.class);

    private int data[];
    private int index = 0;
    private boolean firstpass = true;
    private boolean lastBool = false; // for breaking ties with bools.

    public static void main(String args[]){
        SimpleLowPass lo = new SimpleLowPass();
        lo.data = new int[10];

//        boolean[] samples = {true,true,true,true,true,true,false,false,true,true,false,true,false,true};
//        boolean[] samples = {true,false,true,false,true,false,true,false,true,true,false,false,false,false};
        byte[] samples = {(byte)100,(byte)100,(byte)100,(byte)0,(byte)-1};
        
        for (int i = 0; i < samples.length; i++)
        {
            System.out.println(lo.filter(samples[i]));
        }
    }

    @Override
    public void configure(ParameterMap params) {
        data = new int[params.getRequiredInt("samples")];
        logger.info("\t\tconfigured SimpleLowPass filter for " + data.length + " samples.");
    }

    @Override
    public byte filter(byte b) {
        // a lot of casting on behalf of generalizing this. maybe don't?
        int average = (int)add((int)b);
        return (byte)average;
    }

    @Override
    public boolean filter(boolean b) {
        // maybe more efficient not to generalize?  for booleans, a simple up
        // down count is enough. 
        // (if the new value is true, add one. otherwise, subtract. then
        //  do the same thing with the erased value from the ring buffer)
        // would avoid calculating an average plus all decimal conversions.
        double average = add(b ? 1 : 0);
        if (average ==.5)
        {
            return lastBool;
        }else if (average < .5){
            lastBool = false;
            return false;
        }else if (average > .5){
            lastBool = true;
            return true;
        }

        return b;
    }

    // ring buffered average
    private double add(int sample){
        data[index++] = sample;
        if (index == data.length){
            index = 0;
            firstpass = false;
        }
        int total = 0;
        int samples = firstpass ? index : data.length;
        for (int i = 0; i < samples; i++){
            total += data[i];
        }
        return total / ((double)samples);
    }
}