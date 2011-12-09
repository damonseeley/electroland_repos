package net.electroland.eio;

import java.util.Collection;
import java.util.List;
import java.util.Vector;

import javax.vecmath.Point3d;

import net.electroland.eio.devices.IODevice;
import net.electroland.eio.filters.IOFilter;

public abstract class IOState {

    protected String id;
    protected Point3d location;
    protected String units;
    protected Vector<IOFilter> filters = new Vector<IOFilter>();

    protected List<String>tags;
    protected IODevice device;

    public IOState(String id, double x, double y, double z, String units)
    {
        this.id = id;
        location = new Point3d(x,y,z);
        this.units = units;
    }
    public Point3d getLocation()
    {
        return location;
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
        sb.append(", location=");
        sb.append(location);
        sb.append(", filters=");
        sb.append(filters);
        sb.append(", tags=");
        sb.append(tags);
        sb.append(", device.name=");
        sb.append(device != null ? device.getName() : null);
        sb.append('}');
        return sb.toString();
    }
}