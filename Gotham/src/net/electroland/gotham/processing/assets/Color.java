package net.electroland.gotham.processing.assets;

public class Color {

    public float r, g, b;

    public Color(float r, float g, float b){
        this.r = r;
        this.g = g;
        this.b = b;
    }

    public String toString(){
    	StringBuffer sb = new StringBuffer("Color[");
    	sb.append("r=").append(r).append(", ");
    	sb.append("g=").append(g).append(", ");
    	sb.append("b=").append(b);
    	sb.append(']');
    	return sb.toString();
    }
}