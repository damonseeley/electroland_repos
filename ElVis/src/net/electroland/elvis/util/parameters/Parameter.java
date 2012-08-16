package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public abstract class Parameter {
	ElProps props;
	String name;
	
	
	public Parameter(String name) {
		this.name = name;		
	}
	
	public String getName() {
		return name;
	}

	public abstract int getIntValue();
	public abstract float getFloatValue();
	public abstract double getDoubleValue();
	public abstract boolean getBoolValue();
	
	public abstract void inc();
	public abstract void dec();
	public abstract void setValue(int v);
	public abstract void setValue(float v);
	public abstract void setValue(double v);
	public abstract void setValue(boolean v);
	public abstract void writeToProps(ElProps props);

}
