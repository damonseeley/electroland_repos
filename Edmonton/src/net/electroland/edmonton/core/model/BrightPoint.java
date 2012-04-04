package net.electroland.edmonton.core.model;

public class BrightPoint{
    public double x;
    public int brightness;
    public boolean playSound = false;
    
    public BrightPoint(double _x, int _b)
    {
        this.x = _x;
        this.brightness = _b;
    }
}