package net.electroland.elvis.regions;

import java.util.StringTokenizer;

import net.electroland.elvis.net.StringAppender;

public class BasePolyRegion implements StringAppender {
	public int id;
	public boolean isTriggered;// = false;
	public double mean; //= -1;
	public String name; //= "Region_" + id;
	public float percentage = .5f;


	public BasePolyRegion(int id, boolean isTrig, double m, String name, float percentage) {
		this.id = id;
		isTriggered = isTrig;
		mean = m;
		this.name = name;
		this.percentage = percentage;
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
		sb.append(percentage);
		sb.append(",");
	}


	public static BasePolyRegion buildFromTokenizer(StringTokenizer t) {
		if (! t.hasMoreTokens()) {
			return null;
		}
		int id = Integer.parseInt(t.nextToken());
		String name = t.nextToken();
		Boolean trig = Boolean.parseBoolean(t.nextToken());
		Double d = Double.parseDouble(t.nextToken());
		float p = Float.parseFloat(t.nextToken());
		return new BasePolyRegion(id, trig, d, name, p);
	}


}
