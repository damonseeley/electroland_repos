package net.electroland.eio.filters;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class BoxcarFilter implements IOFilter {

    private static Logger logger = Logger.getLogger(BoxcarFilter.class);

    private int data[];
    private int index = 0;
    private double total = 0;
    // minor bug: the total for the byte filter should default to 0.  The
    // total for the boolean filter should default to data.length * -1. It's no
    // big deal since it fixes itself after the first cycle.

    public static void main(String args[]){

        BoxcarFilter bx1 = new BoxcarFilter();
        BoxcarFilter bx2 = new BoxcarFilter();
        BoxcarFilter bx3 = new BoxcarFilter();

        // since we can't call configure...
        bx1.data = new int[10];
        bx2.data = new int[10];
        bx3.data = new int[10];

      boolean[] samples = {true,true,true,true,true,true,false,false,true,true,false,true,false,true};
//        boolean[] samples = {true,false,false,false,true,false,true,false,true,true,false,false,false,false,
//              true,false,true,false,true,false,true,false,true,true,false,false,false,false};
//      byte[] samples = {(byte)100,(byte)100,(byte)100,(byte)0,(byte)-1};
        
        for (int i = 0; i < samples.length; i++)
        {
            System.out.println("1: " + bx1.filter(samples[i]));
            System.out.println("2: " + bx2.filter(bx3.filter(samples[i])));
        }
    }

    @Override
    public void configure(ParameterMap params) {
        data = new int[params.getRequiredInt("samples")];
        logger.info("\t\tconfigured BoxcarFilter for " + data.length + " samples.");
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