package net.electroland.gotham.core.ui;

public interface ControlBarListener {
    public void allOn();
    public void allOff();
    public void start();
    public void stop();
    public void changeDisplay(String display);
    public void run(String runner);
}