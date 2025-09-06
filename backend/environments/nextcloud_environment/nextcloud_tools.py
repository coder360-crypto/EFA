"""
Nextcloud Tools

Implementation of Nextcloud-specific tools for file management,
sharing, and other Nextcloud operations.
"""

from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin
import os
import json
from datetime import datetime


class NextcloudTools:
    """Nextcloud tools implementation"""
    
    def __init__(self, base_url: str, session: aiohttp.ClientSession, logger: logging.Logger):
        self.base_url = base_url
        self.session = session
        self.logger = logger
        
        # WebDAV and OCS API endpoints
        self.webdav_endpoint = f"{base_url}/remote.php/dav/files"
        self.ocs_endpoint = f"{base_url}/ocs/v2.php"
    
    async def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files and directories in a Nextcloud path
        
        Args:
            path: Path to list (default: root directory)
            
        Returns:
            List of files and directories with their properties
        """
        try:
            # Get username from session auth
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{path}"
            
            # WebDAV PROPFIND request
            headers = {
                "Depth": "1",
                "Content-Type": "application/xml"
            }
            
            propfind_body = '''<?xml version="1.0"?>
            <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
                <d:prop>
                    <d:displayname/>
                    <d:getcontentlength/>
                    <d:getcontenttype/>
                    <d:getlastmodified/>
                    <d:resourcetype/>
                    <oc:size/>
                    <oc:permissions/>
                </d:prop>
            </d:propfind>'''
            
            async with self.session.request(
                "PROPFIND", url, headers=headers, data=propfind_body
            ) as response:
                if response.status not in [200, 207]:
                    raise Exception(f"Failed to list files: HTTP {response.status}")
                
                xml_content = await response.text()
                return self._parse_webdav_response(xml_content)
                
        except Exception as e:
            self.logger.error(f"Failed to list files in '{path}': {e}")
            raise
    
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a file to Nextcloud
        
        Args:
            local_path: Local file path to upload
            remote_path: Remote path where file should be uploaded
            overwrite: Whether to overwrite existing file
            
        Returns:
            Upload result information
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{remote_path}"
            
            # Check if file exists and handle overwrite
            if not overwrite:
                try:
                    await self.get_file_info(remote_path)
                    raise Exception(f"File already exists: {remote_path}")
                except Exception as e:
                    if "not found" not in str(e).lower():
                        raise
            
            # Upload file
            with open(local_path, 'rb') as file:
                async with self.session.put(url, data=file) as response:
                    if response.status not in [200, 201, 204]:
                        raise Exception(f"Upload failed: HTTP {response.status}")
                    
                    return {
                        "success": True,
                        "local_path": local_path,
                        "remote_path": remote_path,
                        "size": os.path.getsize(local_path),
                        "uploaded_at": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to upload file '{local_path}' to '{remote_path}': {e}")
            raise
    
    async def download_file(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """
        Download a file from Nextcloud
        
        Args:
            remote_path: Remote file path to download
            local_path: Local path where file should be saved
            
        Returns:
            Download result information
        """
        try:
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{remote_path}"
            
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Download failed: HTTP {response.status}")
                
                with open(local_path, 'wb') as file:
                    async for chunk in response.content.iter_chunked(8192):
                        file.write(chunk)
                
                return {
                    "success": True,
                    "remote_path": remote_path,
                    "local_path": local_path,
                    "size": os.path.getsize(local_path),
                    "downloaded_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to download file '{remote_path}' to '{local_path}': {e}")
            raise
    
    async def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file or directory from Nextcloud
        
        Args:
            path: Path to delete
            
        Returns:
            Deletion result information
        """
        try:
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{path}"
            
            async with self.session.delete(url) as response:
                if response.status not in [200, 204]:
                    raise Exception(f"Delete failed: HTTP {response.status}")
                
                return {
                    "success": True,
                    "path": path,
                    "deleted_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to delete '{path}': {e}")
            raise
    
    async def create_directory(self, path: str) -> Dict[str, Any]:
        """
        Create a directory in Nextcloud
        
        Args:
            path: Directory path to create
            
        Returns:
            Creation result information
        """
        try:
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{path}"
            
            async with self.session.request("MKCOL", url) as response:
                if response.status not in [200, 201]:
                    raise Exception(f"Directory creation failed: HTTP {response.status}")
                
                return {
                    "success": True,
                    "path": path,
                    "created_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to create directory '{path}': {e}")
            raise
    
    async def move_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """
        Move or rename a file/directory in Nextcloud
        
        Args:
            source_path: Source path
            destination_path: Destination path
            
        Returns:
            Move result information
        """
        try:
            username = self.session._default_auth.login
            source_url = f"{self.webdav_endpoint}/{username}{source_path}"
            destination_url = f"{self.webdav_endpoint}/{username}{destination_path}"
            
            headers = {
                "Destination": destination_url
            }
            
            async with self.session.request("MOVE", source_url, headers=headers) as response:
                if response.status not in [200, 201, 204]:
                    raise Exception(f"Move failed: HTTP {response.status}")
                
                return {
                    "success": True,
                    "source_path": source_path,
                    "destination_path": destination_path,
                    "moved_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to move '{source_path}' to '{destination_path}': {e}")
            raise
    
    async def copy_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """
        Copy a file/directory in Nextcloud
        
        Args:
            source_path: Source path
            destination_path: Destination path
            
        Returns:
            Copy result information
        """
        try:
            username = self.session._default_auth.login
            source_url = f"{self.webdav_endpoint}/{username}{source_path}"
            destination_url = f"{self.webdav_endpoint}/{username}{destination_path}"
            
            headers = {
                "Destination": destination_url
            }
            
            async with self.session.request("COPY", source_url, headers=headers) as response:
                if response.status not in [200, 201, 204]:
                    raise Exception(f"Copy failed: HTTP {response.status}")
                
                return {
                    "success": True,
                    "source_path": source_path,
                    "destination_path": destination_path,
                    "copied_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to copy '{source_path}' to '{destination_path}': {e}")
            raise
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a file or directory
        
        Args:
            path: Path to get information about
            
        Returns:
            File information
        """
        try:
            username = self.session._default_auth.login
            url = f"{self.webdav_endpoint}/{username}{path}"
            
            headers = {
                "Depth": "0",
                "Content-Type": "application/xml"
            }
            
            propfind_body = '''<?xml version="1.0"?>
            <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
                <d:prop>
                    <d:displayname/>
                    <d:getcontentlength/>
                    <d:getcontenttype/>
                    <d:getlastmodified/>
                    <d:resourcetype/>
                    <oc:size/>
                    <oc:permissions/>
                    <oc:id/>
                </d:prop>
            </d:propfind>'''
            
            async with self.session.request(
                "PROPFIND", url, headers=headers, data=propfind_body
            ) as response:
                if response.status not in [200, 207]:
                    raise Exception(f"File not found: HTTP {response.status}")
                
                xml_content = await response.text()
                files = self._parse_webdav_response(xml_content)
                
                if not files:
                    raise Exception("File not found")
                
                return files[0]  # Return first (and only) result
                
        except Exception as e:
            self.logger.error(f"Failed to get file info for '{path}': {e}")
            raise
    
    async def share_file(
        self,
        path: str,
        password: Optional[str] = None,
        expire_date: Optional[str] = None,
        permissions: int = 1
    ) -> Dict[str, Any]:
        """
        Create a share link for a file or directory
        
        Args:
            path: Path to share
            password: Optional password for the share
            expire_date: Optional expiration date (YYYY-MM-DD)
            permissions: Share permissions (1=read, 2=update, 4=create, 8=delete, 16=share)
            
        Returns:
            Share information including URL
        """
        try:
            url = f"{self.ocs_endpoint}/apps/files_sharing/api/v1/shares"
            
            data = {
                "path": path,
                "shareType": 3,  # Public link
                "permissions": permissions
            }
            
            if password:
                data["password"] = password
            
            if expire_date:
                data["expireDate"] = expire_date
            
            headers = {
                "OCS-APIRequest": "true",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Share creation failed: HTTP {response.status}")
                
                response_text = await response.text()
                # Parse XML response from OCS API
                root = ET.fromstring(response_text)
                
                # Check for errors
                status_code = root.find(".//statuscode")
                if status_code is not None and status_code.text != "100":
                    message = root.find(".//message")
                    error_msg = message.text if message is not None else "Unknown error"
                    raise Exception(f"Share creation failed: {error_msg}")
                
                # Extract share information
                data_elem = root.find(".//data")
                if data_elem is None:
                    raise Exception("Invalid response format")
                
                share_info = {}
                for child in data_elem:
                    share_info[child.tag] = child.text
                
                return {
                    "success": True,
                    "path": path,
                    "share_url": share_info.get("url"),
                    "share_id": share_info.get("id"),
                    "token": share_info.get("token"),
                    "permissions": permissions,
                    "created_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to share '{path}': {e}")
            raise
    
    async def search_files(self, query: str, path: str = "/") -> List[Dict[str, Any]]:
        """
        Search for files in Nextcloud
        
        Args:
            query: Search query
            path: Path to search in (default: root)
            
        Returns:
            List of matching files
        """
        try:
            # This is a simplified search implementation
            # In a real implementation, you might use Nextcloud's search API
            
            # Get all files in the specified path
            all_files = await self.list_files(path)
            
            # Filter files by query (simple case-insensitive match)
            matching_files = []
            query_lower = query.lower()
            
            for file_info in all_files:
                name = file_info.get("name", "").lower()
                if query_lower in name:
                    matching_files.append(file_info)
            
            # Recursively search subdirectories (limited depth to avoid infinite loops)
            for file_info in all_files:
                if file_info.get("type") == "directory" and file_info.get("name") not in [".", ".."]:
                    subdir_path = f"{path.rstrip('/')}/{file_info['name']}"
                    try:
                        subdir_results = await self.search_files(query, subdir_path)
                        matching_files.extend(subdir_results)
                    except Exception:
                        # Skip subdirectories that can't be accessed
                        continue
            
            return matching_files
            
        except Exception as e:
            self.logger.error(f"Failed to search files with query '{query}' in '{path}': {e}")
            raise
    
    def _parse_webdav_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse WebDAV XML response into a list of file information
        
        Args:
            xml_content: XML content from WebDAV response
            
        Returns:
            List of file information dictionaries
        """
        try:
            root = ET.fromstring(xml_content)
            files = []
            
            # Define namespaces
            namespaces = {
                'd': 'DAV:',
                'oc': 'http://owncloud.org/ns'
            }
            
            for response in root.findall('.//d:response', namespaces):
                href = response.find('d:href', namespaces)
                propstat = response.find('d:propstat', namespaces)
                
                if href is None or propstat is None:
                    continue
                
                # Extract file path from href
                file_path = href.text
                file_name = os.path.basename(file_path.rstrip('/'))
                
                # Skip the parent directory entry
                if not file_name:
                    continue
                
                prop = propstat.find('d:prop', namespaces)
                if prop is None:
                    continue
                
                # Extract properties
                file_info = {
                    "name": file_name,
                    "path": file_path,
                    "type": "file"
                }
                
                # Check if it's a directory
                resourcetype = prop.find('d:resourcetype', namespaces)
                if resourcetype is not None and resourcetype.find('d:collection', namespaces) is not None:
                    file_info["type"] = "directory"
                
                # Extract other properties
                displayname = prop.find('d:displayname', namespaces)
                if displayname is not None:
                    file_info["display_name"] = displayname.text
                
                contentlength = prop.find('d:getcontentlength', namespaces)
                if contentlength is not None:
                    file_info["size"] = int(contentlength.text) if contentlength.text else 0
                
                contenttype = prop.find('d:getcontenttype', namespaces)
                if contenttype is not None:
                    file_info["content_type"] = contenttype.text
                
                lastmodified = prop.find('d:getlastmodified', namespaces)
                if lastmodified is not None:
                    file_info["last_modified"] = lastmodified.text
                
                # Nextcloud-specific properties
                oc_size = prop.find('oc:size', namespaces)
                if oc_size is not None:
                    file_info["oc_size"] = int(oc_size.text) if oc_size.text else 0
                
                oc_permissions = prop.find('oc:permissions', namespaces)
                if oc_permissions is not None:
                    file_info["permissions"] = oc_permissions.text
                
                oc_id = prop.find('oc:id', namespaces)
                if oc_id is not None:
                    file_info["id"] = oc_id.text
                
                files.append(file_info)
            
            return files
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse WebDAV XML response: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error processing WebDAV response: {e}")
            return []
