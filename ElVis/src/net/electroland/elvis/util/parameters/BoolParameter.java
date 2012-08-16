package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public class BoolParameter extends Parameter {
	
	boolean value;

	public BoolParameter(String name, boolean defvalue,  ElProps props) {
		this(name, props.getProperty(name, defvalue));
	}

	public BoolParameter(String name, boolean value) {
		super(name);
		setValue(value);
	}


	@Override
	public int getIntValue() {
		return value ? 1 : 0;
	}

	@Override
	public float getFloatValue() {
		return value ? 1 : 0;
	}

	@Override
	public void inc() {
		value = ! value;
	}
	public void dec() {
		value = ! value;
	}

	@Override
	public void setValue(int v) {
		value  = v != 0;
	}

	@Override
	public void setValue(float v) {
		value  = v != 0;
	}

	@Override
	public double getDoubleValue() {
		return value ? 1 : 0;
	}

	@Override
	public void setValue(double v) {
		value  = v != 0;

	}
	public  void writeToProps(ElProps props) {
		props.setProperty(name, value);
	}

	@Override
	public boolean getBoolValue() {
		return value;
	}

	@Override
	public void setValue(boolean v) {
		value = v;
	}


}
