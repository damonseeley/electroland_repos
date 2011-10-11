package net.electroland.eio;

import java.util.Collection;
import java.util.Vector;

import net.electroland.eio.filters.IOFilter;

public abstract class IOState {

    protected String id;
    protected Vector<IOFilter> filters = new Vector<IOFilter>();
    protected int x,y,z;
    protected String units;

    public IOState(String id, int x, int y, int z, String units)
    {
        this.id = id;
        this.x = x;
        this.y = y;
        this.z = z;
        this.units = units;
    }

    public int getX() {
        return x;
    }
    public int getY() {
        return y;
    }
    public int getZ() {
        return z;
    }
    public String getUnits() {
        return units;
    }
    final public String getID()
    {
        return id;
    }
    final protected Collection<IOFilter> getFilters()
    {
        return filters;
    }
    final protected void setID(String id)
    {
        this.id = id;
    }
    final protected void addFilter(IOFilter filter)
    {
        filters.add(filter);
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer("{id=");
        sb.append(id);
        sb.append(", x=");
        sb.append(x);
        sb.append(", y=");
        sb.append(y);
        sb.append(", z=");
        sb.append(z);
        sb.append(", filters=");
        sb.append(filters);
        sb.append('}');
        return sb.toString();
    }
}