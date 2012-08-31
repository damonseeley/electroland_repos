package net.electroland.elvis.regions;

import java.util.StringTokenizer;

import net.electroland.elvis.net.StringAppender;

public class BasePolyReagion implements StringAppender {
	public int id;
	public boolean isTriggered;// = false;
	public double mean; //= -1;
	public String name; //= "Region_" + id;


	public BasePolyReagion(int id, boolean isTrig, double m, String name) {
		this.id = id;
		isTriggered = isTrig;
		mean = m;
		this.name = name;
	}


	@Override
	public void buildString(StringBuilder sb) {
		sb.append(id);
		sb.append(",");
		sb.append(name);
		sb.append(",");
		sb.append(isTriggered);
		sb.append(",");
		sb.append(mean);
		sb.append(",");
	}


	public static BasePolyReagion buildFromTokenizer(StringTokenizer t) {
		if (! t.hasMoreTokens()) {
			return null;
		}
		int id = Integer.parseInt(t.nextToken());
		String name = t.nextToken();
		Boolean trig = Boolean.parseBoolean(t.nextToken());
		Double d = Double.parseDouble(t.nextToken());
		return new BasePolyReagion(id, trig, d, name);
	}


}
