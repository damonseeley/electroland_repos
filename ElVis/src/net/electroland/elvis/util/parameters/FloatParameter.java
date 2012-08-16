package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public class FloatParameter extends Parameter {
	float value;
	float incAmount;
	float minValue;
	
	public FloatParameter(String name, float incAmount, float defvalue, ElProps props) {
		this(name, incAmount, props.getProperty(name, defvalue));
	}
	
	public FloatParameter(String name, float incAmount, float value) {
		super(name);
		setValue(value);
		this.incAmount = incAmount;
	}

	public void setMinValue (int minValue) {
		this.minValue = minValue;
	}
	
	@Override
	public int getIntValue() {
		return (int) value;
	}

	@Override
	public float getFloatValue() {
		return value;
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
		setValue((float)v);
	}

	@Override
	public void setValue(float v) {
		value = (v < minValue) ? minValue : v;
	}

	@Override
	public double getDoubleValue() {
		return value;
	}

	@Override
	public void setValue(double v) {
		setValue( Double.valueOf(v).floatValue());
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
		float f = v ? 1: 0;
		setValue(f);
	}

}
