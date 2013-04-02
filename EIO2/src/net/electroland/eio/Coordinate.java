package net.electroland.eio;


public class Coordinate {

    private float x,y,z;
    private String units;

    public Coordinate(float x, float y, float z, String units){
        this.x     = x;
        this.y     = y;
        this.z     = z;
        this.units = units;
    }

    public float getX() {
        return x;
    }

    public float getY() {
        return y;
    }

    public float getZ() {
        return z;
    }

    public String getUnits() {
        return units;
    }

    public String toString(){
        return "Coordinate[Point(" + x + ", " + y + ", " + z + "), units=" + units + "]";
    }
}