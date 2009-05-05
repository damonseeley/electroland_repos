package net.electroland.broadcast.server;

import junit.framework.TestCase;

public class FlexBroadcasterUtilTest extends TestCase {

	public void testGetUniqueId() {
		for (int i = 0; i < 100; i++){
			System.out.println(FlexBroadcasterUtil.getUniqueId());
		}
	}
}