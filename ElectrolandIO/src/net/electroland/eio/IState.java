package net.electroland.eio;

import net.electroland.eio.filters.IOFilter;

public class IState extends IOState{

    boolean state;
    
    public IState(String id, int x, int y, int z, String units) {
        super(id, x, y, z, units);
    }
    public void setState(boolean state)
    {
        for (IOFilter f : this.filters)
        {
            state = f.filter(state);
        }
        this.state = state;
        if (this.state){
            System.out.println("ID: " + id + " is ON");
        }
    }
    public boolean getState()
    {
        return state;
    }
}
