package net.electroland.eio;

public class IOManagertest {

    /**
     * @param args
     */
    public static void main(String[] args) {
        IOManager iom = new IOManager();
        iom.init();
        System.out.println(iom.getStates());
        System.out.println(iom.getStatesForDevice("phoenix0"));
        System.out.println(iom.getStatesForTag("2"));
        System.out.println(iom.getStateById("192.168.247.23-6"));
    }

}
