package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class AddOffset implements Filter {

    private double amount;
    
    @Override
    public void configure(ParameterMap map) {
        amount = map.getRequiredDouble("amount");
    }

    @Override
    public void filter(Value in) {
    	in.setValue((short) (in.getValue() + amount));
    }

}
