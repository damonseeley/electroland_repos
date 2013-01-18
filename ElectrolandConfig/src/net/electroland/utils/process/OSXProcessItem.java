package net.electroland.utils.process;

public class OSXProcessItem extends ProcessItem {

    public OSXProcessItem(String unparsedItem){
        System.out.println("osx: " + unparsedItem);
    }

    @Override
    public boolean equals(Object another) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public int getPID() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public String getName() {
        // TODO Auto-generated method stub
        return null;
    }

}
