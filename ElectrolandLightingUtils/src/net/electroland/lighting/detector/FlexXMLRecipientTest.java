package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.net.UnknownHostException;

import org.junit.Test;

public class FlexXMLRecipientTest {

	byte data[] = {(byte)0,(byte)255,(byte)128,(byte)-1};
	@Test
	public void testSend() throws UnknownHostException {
		FlexXMLRecipient tester = new FlexXMLRecipient("my recipient", 10000, 4, new Dimension(1,1));
		tester.addMessage("hello 1");
		tester.addMessage("hello 2");
		tester.send(data);
		tester.send(data);
		tester.send(new byte[0]);
		tester.addMessage("hello 1");
		tester.addMessage("hello 2");
		tester.send(new byte[0]);
	}
}