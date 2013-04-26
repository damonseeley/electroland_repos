package net.electroland.norfolk.eio.filters;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class AbsoluteValueFilter implements Filter {

    @Override
    public void configure(ParameterMap map) {
    }

    @Override
    public void filter(Value in) {
        in.setValue( Math.abs(in.getValue()) );
    }
}