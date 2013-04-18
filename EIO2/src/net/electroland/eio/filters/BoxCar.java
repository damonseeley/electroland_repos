package net.electroland.eio.filters;

import net.electroland.eio.Value;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class BoxCar implements Filter {

    private static Logger logger = Logger.getLogger(BoxCar.class);

    private int     data[];
    private int     index = 0;
    private double  total = 0;

    // unit test
    public static void main(String args[]){

        // TODO: write test.

    }

    @Override
    public void configure(ParameterMap params) {
        data = new int[params.getRequiredInt("samples")];
        logger.info("\t\tconfigured BoxcarFilter for " + data.length + " samples.");
    }

    @Override
    public void filter(Value in) {
        add(in.getValue());
        in.setValue(getAverage());
    }

    private int getAverage(){
        return (int)(total/data.length);
    }

    // ring buffered sum
    private void add(int sample){

        total -= data[index];
        total += sample;
        data[index++] = sample;

        if (index == data.length){
            index = 0;
        }
    }
}