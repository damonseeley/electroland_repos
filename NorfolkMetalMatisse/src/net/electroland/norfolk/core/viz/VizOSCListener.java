package net.electroland.norfolk.core.viz;

import java.awt.Color;

public interface VizOSCListener {
    public void setSensorState(String id, boolean isOn);
    public void setLightColor(String id, Color color);
}
