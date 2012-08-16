package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public class DoubleParameter extends Parameter {
	double value;
	double incAmount;
	
	public DoubleParameter(String name, double incAmount, double defvalue, ElProps props) {
		this(name, incAmount, props.getProperty(name, defvalue));
	}

	
	public DoubleParameter(String name, double incAmount, double value) {
		super(name);
		this.value = value;
		this.incAmount = incAmount;
	}

	@Override
	public int getIntValue() {
		return (int) value;
	}

	@Override
	public float getFloatValue() {
		return  Double.valueOf(value).floatValue();
	}

	@Override
	public void inc() {
		value += incAmount;

	}
	public void dec() {
		value -= incAmount;

	}

	@Override
	public void setValue(int v) {
		value = v;
	}

	@Override
	public void setValue(float v) {
		value = v;
	}

	@Override
	public double getDoubleValue() {
		return value;
	}

	@Override
	public void setValue(double v) {
		value = v;
		
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
