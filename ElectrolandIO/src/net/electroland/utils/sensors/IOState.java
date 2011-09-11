package net.electroland.utils.sensors;

import java.util.Collection;
import java.util.Vector;

import net.electroland.utils.sensors.filters.IOFilter;

public abstract class IOState {

	private String id;
	private Vector<IOFilter> filters = new Vector<IOFilter>();
	private Vector<String> tags = new Vector<String>();
	
	final public String getID()
	{
		return id;
	}
	final protected Collection<IOFilter> getFilters()
	{
		return filters;
	}
	final public Collection<String> getTags()
	{
		return tags;
	}

	final protected void setID(String id)
	{
		this.id = id;
	}
	final protected void addFilter(IOFilter filter)
	{
		filters.add(filter);
	}
	final protected void addTag(String tag)
	{
		tags.add(tag);
	}
}