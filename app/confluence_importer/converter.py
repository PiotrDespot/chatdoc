import os
import bs4
from markdownify import MarkdownConverter


class Converter:
    def __init__(self, out_dir, attachment_folder_name="attachments"):
        # Constructor code
        self.__out_dir = out_dir
        self.attachment_folder_name = attachment_folder_name


    def recurse_findfiles(self, path):
        # Recursive file scanning logic
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                yield from self.recurse_findfiles(entry.path)
            elif entry.is_file(follow_symlinks=False):
                yield entry
            else:
                raise NotImplemented()

    def __convert_images(self, soup):
        for image in soup.find_all("ac:image"):
            url = None
            for child in image.children:
                url = child.get("ri:filename", None)
                break

            if url is None:
                # no URL found for ac:image
                continue

            # Construct new, actually valid HTML tag for images
            srcurl = os.path.join(self.attachment_folder_name, url)
            imgtag = soup.new_tag("img", attrs={"src": srcurl, "alt": srcurl})

            # Insert a line break after the original "ac:image" tag, then replace with an actual img tag
            image.insert_after(soup.new_tag("br"))
            image.replace_with(imgtag)

    @staticmethod
    def __convert_code_blocks(soup):
        for code_macro in soup.find_all("ac:structured-macro", {"ac:name": "code"}):
            language = code_macro.find("ac:parameter", {"ac:name": "language"})
            code_body = code_macro.find("ac:plain-text-body")

            if language and code_body:
                language = language.text
                code_content = code_body.text.strip()

                # Create a new Markdown code block with the appropriate language specifier
                code_block = soup.new_tag("code")
                code_block.string = f"```{language}\n{code_content}\n```\n"  # Append a newline

                # Insert a line break after the code block
                code_macro.insert_after(soup.new_tag("br"))

                # Replace the Confluence code block with the Markdown code block
                code_macro.replace_with(code_block)

    def __convert_atlassian_html(self, soup):
        # Convert Atlassian images to Markdown images
        self.__convert_images(soup)

        # Convert Confluence code blocks to Markdown code blocks
        self.__convert_code_blocks(soup)

        return soup

    def convert(self):
        # Main conversion logic
        for entry in self.recurse_findfiles(self.__out_dir):
            path = entry.path

            if not path.endswith(".html"):
                continue

            print("Converting {}".format(path))
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()

            soup_raw = bs4.BeautifulSoup(data, 'html.parser')
            soup = self.__convert_atlassian_html(soup_raw)

            md = MarkdownConverter().convert_soup(soup)
            newname = os.path.splitext(path)[0]
            with open(newname + ".md", "w", encoding="utf-8") as f:
                f.write(md)
