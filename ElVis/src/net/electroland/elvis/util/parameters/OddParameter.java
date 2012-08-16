package net.electroland.elvis.util.parameters;

import net.electroland.elvis.util.ElProps;

public class OddParameter extends IntParameter {
	
	public OddParameter(String name, int incAmount, int value, ElProps props) {
		super(name, incAmount, value, props);
		if(incAmount % 2 != 0) {
			incAmount+=1;
			System.out.println("incAmound for parameter " + name + " must be even setting to " + incAmount);
		}
		this.incAmount = incAmount;
		
	}


	@Override
	public void setValue(int v) {
		if(v %2 == 0) {
			v +=1;
			System.out.println("value for parameter " + name + " must be odd setting to " + v);
		}
		super.setValue(v);
	}


}
