package net.electroland.eio.filters;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class BoxcarFilter implements IOFilter {

    private static Logger logger = Logger.getLogger(IOFilter.class);

    private boolean firstpass = true; // could probably do without this.
    private int data[];
    private int index = 0;
    private double total = 0;
    private int samples;

    public static void main(String args[]){

        BoxcarFilter bx = new BoxcarFilter();
        bx.data = new int[10];

//      boolean[] samples = {true,true,true,true,true,true,false,false,true,true,false,true,false,true};
//        boolean[] samples = {true,false,false,false,true,false,true,false,true,true,false,false,false,false,
//              true,false,true,false,true,false,true,false,true,true,false,false,false,false};
        byte[] samples = {(byte)100,(byte)100,(byte)100,(byte)0,(byte)-1};
        
        for (int i = 0; i < samples.length; i++)
        {
            System.out.println(bx.filter(samples[i]));
        }
    }

    @Override
    public void configure(ParameterMap params) {
        data = new int[params.getRequiredInt("samples")];
        logger.info("\t\tconfigured SimpleLowPass filter for " + data.length + " samples.");
    }

    @Override
    public byte filter(byte b) {
        return (byte)add((int)b);
    }

    @Override
    public boolean filter(boolean b) {
        // I removed it, but somewhat more smoothing possible by special
        // casing EQUALS .5 to return whatever the previous eval came to.
        // FWIW, we could skip all this math for the booleans, and just count
        // how many trues and how many falses were in the previous cycles.
        // just can't reusing the code with the byte verions then.
        return add(b ? 1 : 0) > .5;
    }

    // ring buffered average
    private double add(int sample){

        if (!firstpass){
            total -= data[index];
        }else{
            samples = index + 1;
        }

        total += sample;
        data[index++] = sample;

        if (index == data.length){
            index = 0;
            firstpass = false;
        }
        return total / samples;
    }
}