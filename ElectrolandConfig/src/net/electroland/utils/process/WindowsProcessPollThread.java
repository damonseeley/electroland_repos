package net.electroland.utils.process;

import java.io.IOException;
import java.util.List;

public class WindowsProcessPollThread extends ProcessPollThread {

    public WindowsProcessPollThread(ProcessItem item,
            ProcessExitedListener listener, long period) {
        super(item, listener, period);
    }

    @Override
    public boolean processExited() {
        try {

            String cmd = System.getenv("windir") + 
                    "\\system32\\tasklist.exe /v /fi \"PID eq " + 
                    item.getPID() + "\" /fo csv";
            List<String> output = ProcessUtil.getProcessOutput(
                                            Runtime.getRuntime().exec(cmd));

            return output.get(0).startsWith("INFO: No tasks");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}