package net.electroland.utils.process;

public class WindowsProcessItem implements ProcessItem {

    private String name, sessionName, memoryUsage;
    private int PID, sessionNumber;

    public WindowsProcessItem(String unparsedItem){
        // doh. didn't realize until after the fact that tasklist.exe can export TSV (tasklist.exe \fo csv)
        //           1         2         3         4         5         6         7
        // 01234567890123456789012345678901234567890123456789012345678901234567890123456 
        // ========================= ======== ================ =========== ============
        // javaw.exe                     6004 Console                    1     10,640 K
        name            = unparsedItem.substring(0, 25).trim();
        PID             = Integer.parseInt(unparsedItem.substring(26, 34).trim());
        sessionName     = unparsedItem.substring(35, 51).trim();
        sessionNumber   = Integer.parseInt(unparsedItem.substring(52, 63).trim());
        memoryUsage     = unparsedItem.substring(64).trim();
    }

    @Override
    public boolean equals(Object another) {
        if (another instanceof WindowsProcessItem){
            return this.name.equals(((WindowsProcessItem)another).getName()) &&
                    this.PID == ((WindowsProcessItem)another).getPID() &&
                    this.sessionName.equals(((WindowsProcessItem)another).getSessionName()) &&
                    this.sessionNumber == ((WindowsProcessItem)another).getSessionNumber();
        }else{
            return false;
        }
    }

    @Override
    public int getPID() {
        return PID;
    }

    @Override
    public String getName() {
        return name;
    }

    public String getSessionName() {
        return sessionName;
    }

    public int getSessionNumber() {
        return sessionNumber;
    }

    public String getMemoryUsage() {
        return memoryUsage;
    }

    public String toString(){
        StringBuffer sb = new StringBuffer();
        sb.append("WindowsProcessItem[");
        sb.append("name="            + this.getName());
        sb.append(", PID="           + this.getPID());
        sb.append(", sessionName="   + this.getSessionName());
        sb.append(", sessionNumber=" + this.getSessionNumber());
        sb.append(", memoryUsage="   + this.getMemoryUsage());
        sb.append(']');
        return sb.toString();
    }
}