package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public class IntParameter extends Parameter {
	
	int minValue = Integer.MIN_VALUE;
	int value;
	int incAmount;
	
	public IntParameter(String name, int incAmount, int value) {
		super(name);
		setValue(value);
//		this.value = value;
		this.incAmount = incAmount;
	}
	
	public IntParameter(String name, int incAmount, int defvalue, ElProps props) {
		this(name, incAmount, props.getProperty(name, defvalue));
	}

	public void setMinValue (int minValue) {
		this.minValue = minValue;
	}

	@Override
	public int getIntValue() {
		return value;
	}

	@Override
	public float getFloatValue() {
		return (float) value;
	}

	@Override
	public void inc() {
		setValue(value + incAmount);
	}
	public void dec() {
		setValue(value - incAmount);
	}

	@Override
	public void setValue(int v) {
		value = v;
		value = (value < minValue) ? minValue : value;
	}

	@Override
	public void setValue(float v) {
		value = (int) v;
	}

	@Override
	public double getDoubleValue() {
		return value;
	}

	@Override
	public void setValue(double v) {
		value = (int)v;
	}
	public  void writeToProps(ElProps props) {
		props.setProperty(name, value);
	}

	@Override
	public boolean getBoolValue() {
		return value==0;
	}

	@Override
	public void setValue(boolean v) {
		value = v ? 1: 0;
	}


}
