package net.electroland.ea.changes;

import net.electroland.ea.Change;
import net.electroland.ea.State;

public class LinearChange extends Change {

    @Override
    public State nextState(State init, double percentComplete) {
        
        int x = (int)between(init.geometry.x, this.getTargetState(init).geometry.x, percentComplete);
        int y = (int)between(init.geometry.y, this.getTargetState(init).geometry.y, percentComplete);
        int w = (int)between(init.geometry.width, this.getTargetState(init).geometry.width, percentComplete);
        int h = (int)between(init.geometry.height, this.getTargetState(init).geometry.height, percentComplete);
        double a = (float)between(init.alpha, this.getTargetState(init).alpha, percentComplete);
        return new State(x,y,w,h,a);
    }
    private static double between(int start, int finish, double percentComplete)
    {
        if (start == finish) 
            return finish;
        else
            return start + ((finish - start) * percentComplete);
    }
    private static double between(double start, double finish, double percentComplete)
    {
        if (start == finish)
            return finish;
        else
            return (start + ((finish - start) * percentComplete));
    }
}