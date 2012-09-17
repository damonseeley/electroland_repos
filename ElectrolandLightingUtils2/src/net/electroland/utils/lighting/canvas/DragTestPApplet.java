package net.electroland.utils.lighting.canvas;

import processing.event.MouseEvent;

public class DragTestPApplet extends ELUPApplet {

    private static final long serialVersionUID = -3921716170593769069L;
    int x, y;
    int radius = 25;

    @Override
    public void setup() {
        registerMethod("mouseEvent", this);
    }

    @Override
    public void drawELUContent() {
        // erase background
        color(0);
        fill(0);
        rect(0,0,this.getWidth(), this.getHeight());

        fill(255);
        ellipse(x, y, radius, radius);
    }

    public void mouseEvent(MouseEvent event) {
        x = event.getX();
        y = event.getY();

        switch (event.getAction()) {
          case MouseEvent.DRAGGED:
            break;
        }
       }

}
