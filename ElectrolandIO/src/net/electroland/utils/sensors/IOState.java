package net.electroland.utils.sensors;

import java.util.Collection;
import java.util.Vector;

import net.electroland.utils.sensors.filters.IOFilter;

public abstract class IOState {

	protected String id;
	protected Vector<IOFilter> filters = new Vector<IOFilter>();
	protected int x,y,z;
	protected String units;
	
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
}