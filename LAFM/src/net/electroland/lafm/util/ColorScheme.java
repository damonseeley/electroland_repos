package net.electroland.lafm.util;

public class ColorScheme {
	
	/**
	 * Creates a custom color spectrum defined by multiple RGB arrays
	 * positioned at particular points along a range of values from
	 * 0.0 to 1.0.
	 * 
	 * @author Aaron Siegel
	 */
	
	float[][] colorlist;
	float[] pointlist;
	//int above, below;
	//float diff, percent;

	public ColorScheme(float[][] colorlist, float[] pointlist){
		this.colorlist = colorlist;
		this.pointlist = pointlist;
	}
	
	public float[] getColor(float p){
		float[] color = new float[3];
		int above = 0;
		int below = 0;
		if(p > 1){
			p = 0;
		}
		//System.out.println(p);
		for(int i=0; i<pointlist.length; i++){						// for each point in list...
			if(p < pointlist[i]){									// if point is less than current...
				above = i;											// specify above and below positions
		        below = i-1;
		        if(below < 0){
		        	above = 1;
		        	below = 0;
		        }
		        break;
			}
		}
		float percent = map(p, pointlist[below], pointlist[above], 1, 0);	// percent between positions
		//System.out.println(percent);
		color = lerpColor(colorlist[below], colorlist[above], percent);
		return color;
	}
	
	public float map(float value, float oldlow, float oldhigh, float newlow, float newhigh){
		float tempval = (oldhigh - oldlow) - (value - oldlow);		// difference between low and value
		//System.out.println(tempval);
		tempval = tempval/(oldhigh-oldlow);							// normalize value by old range
		//System.out.println(value +" "+ oldhigh-oldlow);
		float newvalue = (tempval * (newhigh - newlow)) + newlow;	// multiply by new range plus low value
		return newvalue;
	}
	
	public float[] lerpColor(float[] oldcolor, float[] newcolor, float value){
		float[] color = new float[3];
		color[0] = ((newcolor[0]-oldcolor[0]) * value) + oldcolor[0];
		color[1] = ((newcolor[1]-oldcolor[1]) * value) + oldcolor[1];
		color[2] = ((newcolor[2]-oldcolor[2]) * value) + oldcolor[2];
		return color;
	}
	
}
