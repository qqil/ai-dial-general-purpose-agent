from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        #TODO:
        # 1. Create instance of MCPClient with `cls`
        # 2. Connect to MCP server
        # 3. return created instance
        instance = cls(mcp_server_url)
        await instance.connect()
        return instance

    async def connect(self):
        """Connect to MCP server"""
        #TODO:
        # 1. Check if session is present, if yes just return to finsh execution
        # 2. Call `streamablehttp_client` method with `server_url` and set as `self._streams_context`
        # 3. Enter `self._streams_context`, result set as `read_stream, write_stream, _`
        # 4. Create ClientSession with streams from above and set as `self._session_context`
        # 5. Enter `self._session_context` and set as self.session
        # 6. Initialize session and print its result to console
        if self.session is not None:
            return
        
        self._streams_context = streamablehttp_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        init_result = await self.session.initialize()

        print(f"MCP Client initialized: {init_result}")


    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        #TODO: Get and return MCP tools as list of MCPToolModel
        if self.session is None:
            raise Exception("MCP Client is not connected. Call connect() before getting tools.")
        tools_data = await self.session.list_tools()

        return [MCPToolModel(
            name=tool.name,
            description=str(tool.description),
            parameters=tool.inputSchema
        ) for tool in tools_data.tools]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        #TODO: Make tool call and return its result. Do it in proper way (it returns array of content and you need to handle it properly)
        if self.session is None:
            raise Exception("MCP Client is not connected. Call connect() before calling tools.")
        
        tool_call_result = await self.session.call_tool(tool_name, tool_args)
        if not tool_call_result.content:
            return None
        
        content = tool_call_result.content[0]
        if isinstance(content, TextContent):
            return content.text
        
        return content

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        #TODO: Get and return resource. Resources can be returned as TextResourceContents and BlobResourceContents, you
        #      need to return resource value (text or blob)
        if self.session is None:
            raise Exception("MCP Client is not connected. Call connect() before getting resources.")
        
        resource_result = await self.session.read_resource(uri)
        content = resource_result.contents[0]
        
        if isinstance(content, TextResourceContents):
            return content.text
        elif isinstance(content, BlobResourceContents):
            return content.blob
        
        raise Exception(f"Unexpected resource result type: {type(resource_result)}")

    async def close(self):
        """Close connection to MCP server"""
        #TODO:
        # 1. Close `self._session_context`
        # 2. Close `self._streams_context`
        # 3. Set session, _session_context and _streams_context as None
        if (self._session_context):
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing MCP session context: {str(e)}")

        if (self._streams_context):
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing MCP streams context: {str(e)}")
        
        self._session_context = None
        self._streams_context = None
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False

