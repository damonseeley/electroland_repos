package net.electroland.gotham.processing;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.core.People;
import net.electroland.utils.lighting.canvas.ELUPApplet;

abstract public class GothamPApplet extends ELUPApplet {

    private static final long serialVersionUID = 1L;
    private People pm;

    public void handle(GridData d) {
        synchronized(pm){
            pm = new People(d);
        }
    }

    public People getPeople(){
        return pm;
    }
}