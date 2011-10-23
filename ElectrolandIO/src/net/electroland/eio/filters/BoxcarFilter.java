package net.electroland.eio.filters;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class BoxcarFilter implements IOFilter {

    private static Logger logger = Logger.getLogger(IOFilter.class);

    private int data[];
    private int index = 0;
    private double total = 0;
    // minor bug: the total for the byte filter should default to 0.  The
    // total for the boolean filter should default to data.length * -1. It's no
    // big deal since it fixes itself after the first cycle.

    public static void main(String args[]){

        BoxcarFilter bx = new BoxcarFilter();
        bx.data = new int[10];

//      boolean[] samples = {true,true,true,true,true,true,false,false,true,true,false,true,false,true};
        boolean[] samples = {true,false,false,false,true,false,true,false,true,true,false,false,false,false,
              true,false,true,false,true,false,true,false,true,true,false,false,false,false};
//        byte[] samples = {(byte)100,(byte)100,(byte)100,(byte)0,(byte)-1};
        
        for (int i = 0; i < samples.length; i++)
        {
            System.out.println(bx.filter(samples[i]));
        }
    }

    @Override
    public void configure(ParameterMap params) {
        data = new int[params.getRequiredInt("samples")];
        for (int i = 0; i < data.length; i++){
            data[i] = 0;
        }
        logger.info("\t\tconfigured BoxCarFilter for " + data.length + " samples.");
    }

    @Override
    public byte filter(byte b) {
        add((int)b);
        return (byte)(total/data.length);
    }

    @Override
    public boolean filter(boolean b)
    {
        add(b ? 1 : -1);
        return total > 0;
    }

    // ring buffered average
    private void add(int sample){

        total -= data[index];
        total += sample;
        data[index++] = sample;

        if (index == data.length){
            index = 0;
        }
    }
}