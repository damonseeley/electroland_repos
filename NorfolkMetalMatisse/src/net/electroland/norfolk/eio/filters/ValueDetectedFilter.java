package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class ValueDetectedFilter implements Filter {

    private long timeout, last;
    private int value;

    @Override
    public void configure(ParameterMap map) {
        timeout = map.getRequiredInt("timeout");
        value   = map.getRequiredInt("value");
        last    = 0;
    }

    @Override
    public void filter(Value in) {
        if (System.currentTimeMillis() - last > timeout){
            if (in.getValue() == value){
                last = System.currentTimeMillis();
                in.setValue(Short.MAX_VALUE);
                System.out.println("PERSON ENTERED AT " + last);
            }else{
                in.setValue(0);
            }
        }else{
            in.setValue(0);
        }
    }
}
