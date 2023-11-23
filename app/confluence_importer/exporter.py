import json
import os
import re
import requests
from atlassian import Confluence
from urllib.parse import urlparse, urlunparse


# Constants and Exception classes


class ExportException(Exception):
    pass


class Exporter:
    DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB, since we're single threaded this is safe to raise much higher
    ATTACHMENT_FOLDER_NAME = "attachments"

    def __init__(self, url, username, token, out_dir, space, no_attach):
        # Constructor code
        self.__out_dir = out_dir
        self.__parsed_url = urlparse(url)
        self.__username = username
        self.__token = token
        self.__confluence = Confluence(url=urlunparse(self.__parsed_url),
                                       username=self.__username,
                                       token=self.__token)
        self.__seen = set()
        self.__no_attach = no_attach
        self.__space = space

    @staticmethod
    def __sanitize_filename(document_name_raw):
        # Replace invalid characters with underscores using regular expressions
        sanitized_name = re.sub(r'[\/:*?"<>|]', '_', document_name_raw)
        return sanitized_name

    def __download_attachments(self, page_id, page_output_dir, page_location):
        ret = self.__confluence.get_attachments_from_content(page_id, start=0, limit=500, expand=None,
                                                             filename=None, media_type=None)
        for i in ret["results"]:
            att_title = i["title"]
            download = i["_links"]["download"]

            att_url = urlunparse(
                (self.__parsed_url[0], self.__parsed_url[1], "/wiki/" + download.lstrip("/"), None, None, None)
            )
            att_sanitized_name = self.__sanitize_filename(att_title)
            att_filename = os.path.join(page_output_dir, self.ATTACHMENT_FOLDER_NAME, att_sanitized_name)

            att_dirname = os.path.dirname(att_filename)
            os.makedirs(att_dirname, exist_ok=True)

            print("Saving attachment {} to {}".format(att_title, page_location))

            r = requests.get(att_url, auth=(self.__username, self.__token), stream=True)
            if 400 <= r.status_code:
                if r.status_code == 404:
                    print("Attachment {} not found (404)!".format(att_url))
                    continue

                # this is a real error, raise it
                r.raise_for_status()

            with open(att_filename, "wb") as f:
                for buf in r.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                    f.write(buf)

    @staticmethod
    def __save_page_content(page_filename, content, metadata):
        try:
            with open(page_filename, "w", encoding="utf-8") as f:
                f.write(content)

            # Save metadata in a separate JSON file
            metadata_filename = page_filename.replace(".html", "_metadata.json")
            with open(metadata_filename, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

        except Exception as e:
            print(f"Error saving content to {page_filename}: {str(e)}")

    def __process_child_pages(self, child_ids, parents, page_title):
        for child_id in child_ids:
            self.__dump_page(child_id, parents=parents + [page_title])

    def __dump_page(self, src_id, parents):
        page = self.__confluence.get_page_by_id(src_id, expand="body.storage")
        page_title = page["title"]
        page_id = page["id"]
        page_link = page["_links"]["base"] + page["_links"]["webui"]
        child_ids = self.__confluence.get_child_id_list(page_id)
        content = page["body"]["storage"]["value"]
        history = self.__confluence.history(page_id)
        last_update_date = history["lastUpdated"]["when"]
        last_update_by = history["lastUpdated"]["by"]["displayName"]
        page_metadata = {
            "page_link": page_link,
            "page_id": page_id,
            "last_update_date": last_update_date,
            "last_update_by": last_update_by,
        }
        extension = ".html"
        if len(child_ids) > 0:
            document_name = "index" + extension
        else:
            document_name = page_title + extension

        sanitized_filename = self.__sanitize_filename(document_name)
        sanitized_parents = list(map(self.__sanitize_filename, parents))

        sanitized_parents = [parent.replace("\\", "_") for parent in sanitized_parents]
        page_location = sanitized_parents + [sanitized_filename]
        page_filename = os.path.join(self.__out_dir, *page_location)

        page_output_dir = os.path.dirname(page_filename)
        os.makedirs(page_output_dir, exist_ok=True)
        print("Saving to {}".format(" / ".join(page_location)))

        self.__save_page_content(page_filename, content, page_metadata)

        if not self.__no_attach:
            self.__download_attachments(page_id, page_output_dir, page_location)

        self.__seen.add(page_id)

        for child_id in child_ids:
            self.__dump_page(child_id, parents=sanitized_parents + [page_title])

    def __dump_space(self, space):
        space_key = space["key"]
        print("Processing space", space_key)
        if space.get("homepage") is None:
            print("Skipping space: {}, no homepage found!".format(space_key))
            print("In order for this tool to work there has to be a root page!")
            raise ExportException("No homepage found")
        else:
            homepage_id = space["homepage"]["id"]
            self.__dump_page(homepage_id, parents=[space_key])

    def __dump_all_spaces(self, spaces):
        for space in spaces:
            self.__dump_space(space)

    def __dump_space_by_name(self, space_name):
        # Use the provided space_name to get space information
        space_info = self.__confluence.get_space(space_name, expand='description.plain,homepage')

        if 'key' not in space_info:
            print(f"Space with name '{space_name}' not found.")
            return

        self.__dump_space(space_info)

    def dump(self):
        ret = self.__confluence.get_all_spaces(start=0, limit=500, expand='description.plain,homepage')

        if ret['size'] == 0:
            print("No spaces found in Confluence. Please check credentials")
            return

        print("Available spaces:")
        for i, space in enumerate(ret["results"]):
            print(f"{i + 1}. {space['name']}")

        print("0. Dump all spaces")
        print("-1. Provide your own space name")

        while True:
            try:
                choice = int(
                    input("Enter the number of the space to dump, -1 to provide your own name, or 0 to dump all: "))
                if choice == -1:
                    custom_space_name = input("Enter the name of the space you want to dump: ")
                    self.__dump_space_by_name(custom_space_name)
                    break
                elif choice == 0:
                    self.__dump_all_spaces(ret["results"])
                    break
                elif 1 <= choice <= len(ret["results"]):
                    selected_space = ret["results"][choice - 1]
                    self.__dump_space(selected_space)
                    break
                else:
                    print("Invalid choice. Please enter a valid number, -1, or 0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
