package net.electroland.faces;

import java.awt.Rectangle;

public class Light {

	int universe, channel, color, brightness;
	Rectangle lightbox;

	public Light(int universe, int channel, int x, int y, int width, int height){
		this.universe = universe;
		this.channel = channel;
		this.color = 255;
		this.brightness = 0;// all lights initialize in their "off" state.
		this.lightbox = new Rectangle(x,y,width,height);
	}
	public Light(int universe, int channel, Rectangle lightbox){
		this.universe = universe;
		this.channel = channel;
		this.color = 255;
		this.lightbox = lightbox;
		this.brightness = 0;// all lights initialize in their "off" state.
	}
	public String toString(){
		StringBuffer sb = new StringBuffer("Light[universe=");
		sb.append(universe);
		sb.append(", channel=").append(channel);
		sb.append(", color=").append(color);
		sb.append(", lightbox=[x=").append(lightbox.x);
		sb.append(", y=").append(lightbox.y);
		sb.append(", width=").append(lightbox.width);
		sb.append(", height=").append(lightbox.height);
		sb.append("]]");
		return sb.toString();
	}
	
	public boolean contains(int x1, int y1){
		return lightbox.contains(x1, y1);
	}
}
