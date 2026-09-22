import os
import os.path
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
import hashlib
import joblib
import torch
from datadings.reader import MsgpackReader
from datadings.writer import FileWriter

from ..augmentations.basic import ComposeAug
from pillar.datasets.nlst_utils import get_scaled_annotation_mask, IMG_PAD_TOKEN
import logging


CACHED_FILES_EXT = ".png"
DEFAULT_CACHE_DIR = "default/"

CORUPTED_FILE_ERR = "WARNING! Error processing file from cache - removed file from cache. Error: {}, {}, {}"


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def split_augmentations_by_cache(augmentations):
    """
    Given a list of augmentations, returns a list of tuples. Each tuple
    contains a caching key of the augmentations up to the spiltting point,
    and a list of augmentations that should be applied afterwards.

    split_augmentations will contain all possible splits by cachable augmentations,
    ordered from latest possible one to the former ones.
    The last tuple will have all augmentations.

    Note - splitting will be done for indexes that all augmentations up to them are
    cachable.
    """
    # list of (cache key, post augmentations)
    split_augmentations = []
    split_augmentations.append((DEFAULT_CACHE_DIR, augmentations))
    all_prev_cachable = True
    key = DEFAULT_CACHE_DIR
    for ind, trans in enumerate(augmentations):
        # check trans.cachable() first separately to save run time
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            post_augmentations = augmentations[ind + 1 :] if ind < len(augmentations) else []
            split_augmentations.append((key, post_augmentations))

    return list(reversed(split_augmentations))


def apply_augmentations_and_cache(
    image,
    additional,
    img_path,
    augmentations,
    cache,
    cache_full_size=False,
    cache_last_cachable_only=False,
    base_key="",
):
    """
    Loads the image by its absolute path and apply the augmentations one
    by one (similar to what the composed one is doing).  All first cachable
    transformer's output is cached (until reaching a non cachable one).
    """
    if cache_full_size and not cache.exists(img_path, DEFAULT_CACHE_DIR):
        cache.add(img_path, DEFAULT_CACHE_DIR, image)

    all_prev_cachable = True
    key = base_key

    if len(augmentations):
        last_cachable_index = (
            min(
                [ind for ind, augmentation in enumerate(augmentations) if not augmentation.cachable()],
                default=len(augmentations),
            )
            - 1
        )

        for ind, trans in enumerate(augmentations):
            if additional is not None:
                image = trans(image, **additional)
            else:
                image = trans(image)
            if not all_prev_cachable or not trans.cachable():
                all_prev_cachable = False
            else:
                key += trans.caching_keys()
                if (not cache_last_cachable_only) or (ind == last_cachable_index):
                    cache.add(img_path, key, image)

    return image


class DirectCache:
    def __init__(self, path, extension="", no_cache_misses_allowed=False, compress=True):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension
        self.no_cache_misses_allowed = no_cache_misses_allowed
        self.compress = compress

        if compress:
            if ".z" != extension:
                self.files_extension += ".z"
            elif ".pt" != extension:
                self.files_extension += ".pt"

    def _file_path(self, hashed_key):
        return os.path.join(self.cache_dir, hashed_key + self.files_extension)

    def exists(self, image_path):
        hashed_key = md5(image_path)
        exists = os.path.isfile(self._file_path(hashed_key))
        return exists

    def get(self, image_path, sample=None):
        base_path = os.path.dirname(image_path[0])
        hashed_key = md5(base_path)
        if self.compress:
            image = joblib.load(self._file_path(hashed_key))
        else:
            image = torch.load(self._file_path(hashed_key))

        if isinstance(image, dict):
            # Remove the metadata added for debugging
            if "cache_metadata" in image:
                del image["cache_metadata"]
            for k, v in image.items():
                image[k] = v.to(torch.float32)
        else:
            image = image.to(torch.float32)
        return image

    def add(self, image_path, image):
        base_path = os.path.dirname(image_path[0])
        hashed_key = md5(base_path)
        # Use exist_ok to avoid race conditions
        os.makedirs(self.cache_dir, exist_ok=True)
        if isinstance(image, dict):
            # For debugging purpose
            image["cache_metadata"] = hashed_key
        if self.compress:
            joblib.dump(image, self._file_path(hashed_key), compress=self.compress)
        else:
            torch.save(image, self._file_path(hashed_key))

    def rem(self, image_path):
        hashed_key = md5(image_path)
        try:
            os.remove(self._file_path(hashed_key))
        # Don't raise error if file not exists.
        except OSError:
            pass


class cache:
    def __init__(
        self,
        path,
        extension=CACHED_FILES_EXT,
        no_cache_misses_allowed=False,
        compress=None,
    ):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension
        self.no_cache_misses_allowed = no_cache_misses_allowed
        self.compress = compress

        if compress:
            if ".z" != extension:
                self.files_extension += ".z"
        elif ".pt" != extension:
            self.files_extension += ".pt"

    def _file_dir(self, attr_key):
        return os.path.join(self.cache_dir, attr_key)

    def _file_path(self, attr_key, hashed_key):
        return os.path.join(self.cache_dir, attr_key, hashed_key + self.files_extension)

    def exists(self, image_path, attr_key):
        hashed_key = md5(image_path)
        exists = os.path.isfile(self._file_path(attr_key, hashed_key))
        if self.no_cache_misses_allowed and not exists:
            raise ValueError(f"Cache missed: {image_path}, {hashed_key}, {self._file_path(attr_key, hashed_key)}")
        return exists

    def get(self, image_path, attr_key):
        hashed_key = md5(image_path)
        if self.compress:
            image = joblib.load(self._file_path(attr_key, hashed_key))
        else:
            image = torch.load(self._file_path(attr_key, hashed_key))

        if isinstance(image, dict) and "cache_metadata" in image:
            # Remove the metadata added for debugging
            del image["cache_metadata"]
            for k, v in image.items():
                image[k] = image[k].to(torch.float32)

        return image

    def add(self, image_path, attr_key, image):
        hashed_key = md5(image_path)
        file_dir = self._file_dir(attr_key)
        if self.no_cache_misses_allowed:
            raise ValueError(
                f"Attempting to add cache: {image_path}, {hashed_key}, {self._file_path(attr_key, hashed_key)}"
            )
        # Use exist_ok to avoid race conditions
        os.makedirs(file_dir, exist_ok=True)
        if isinstance(image, dict):
            # For debugging purpose
            image["cache_metadata"] = (attr_key, hashed_key)
        if self.compress:
            joblib.dump(image, self._file_path(attr_key, hashed_key), compress=self.compress)
        else:
            torch.save(image, self._file_path(attr_key, hashed_key))

    def rem(self, image_path, attr_key):
        hashed_key = md5(image_path)
        try:
            os.remove(self._file_path(attr_key, hashed_key))
        # Don't raise error if file not exists.
        except OSError:
            pass


class GroupCache:
    """
    Cache in groups. Convenient and efficient when reading from many dicom files that collectively form a 3D tensor.
    Use [datadings](https://datadings.readthedocs.io/en/latest/index.html) format internally to allow indexing items in a group with keys.
    Only supports storing images in dict. The group key will be the filename of storage. If the group key is not suitable as a filename, please hash the group key.

    The APIs are designed to be close to the original cache API. However, you need to open the group for reading/writing before reading/writing.

    Notes: it's safe to use dataloader workers with group cache, but you need to ensure:
    1. always open the the intended group cache file before writing and close after writing.
    2. ensure only **one worker** writes to a group's cache file.
    3. writing in different threads to the same group at the same time has not been tested.
    """

    def __init__(self, path, extension=CACHED_FILES_EXT, no_cache_misses_allowed=False):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension
        self.no_cache_misses_allowed = no_cache_misses_allowed
        if ".msgpack" != extension:
            self.files_extension += ".msgpack"

        self.readers = {}
        self.writers = {}

    def _group_path(self, attr_key, group_key):
        return os.path.join(self.cache_dir, attr_key, group_key + self.files_extension)

    def exists_group(self, attr_key, group_key, image_path=None):
        # image_path is for making the error message more informative and is optional
        dict_key = (attr_key, group_key)
        if dict_key in self.readers or dict_key in self.writers:
            # It has been opened and thus exists.
            return True

        group_path = self._group_path(attr_key, group_key)
        exists = os.path.isfile(group_path)
        if self.no_cache_misses_allowed and not exists:
            raise ValueError(f"Cache missed: {attr_key}, {group_key}, {group_path} ({image_path})")
        return exists

    def get(self, image_path, attr_key, group_key):
        dict_key = (attr_key, group_key)
        assert dict_key in self.readers, (
            f"Getting {dict_key} for image {image_path} but no reader found, readers: {self.readers}"
        )

        reader = self.readers[dict_key]
        reader = MsgpackReader(self._group_path(attr_key, group_key))
        index = reader.find_index(image_path)
        # Note: get has copy=True by default. Might turn off copying to save memory.
        image = reader.get(index)

        for k in image["tensors"]:
            image[k] = torch.tensor(image[k])
        del image["tensors"]
        del image["key"]
        del image["cache_metadata"]

        return image

    def open_group_for_writing(self, attr_key, group_key):
        dict_key = (attr_key, group_key)
        assert dict_key not in self.writers and dict_key not in self.readers, "The group is already open."
        # Use exist_ok to avoid race conditions
        group_path = self._group_path(attr_key, group_key)
        os.makedirs(os.path.dirname(group_path), exist_ok=True)

        writer = FileWriter(group_path, overwrite=True, disable=True)
        self.writers[dict_key] = writer

    def is_being_written(self, attr_key, group_key):
        dict_key = (attr_key, group_key)
        return dict_key in self.writers

    def close_group(self, attr_key, group_key):
        dict_key = (attr_key, group_key)
        if dict_key in self.writers:
            writer = self.writers[dict_key]

            # might be closed due to removal
            if not writer._outfile.closed:
                writer.close()

            del self.writers[dict_key]

        if dict_key in self.readers:
            reader = self.readers[dict_key]

            reader._close()
            del self.readers[dict_key]

    def close_all_groups(self):
        for writer in self.writers.values():
            # might be closed due to removal
            if not writer._outfile.closed:
                writer.close()

        for reader in self.readers.values():
            reader._close()

        self.writers = {}
        self.readers = {}

    def open_group_for_reading(self, attr_key, group_key, allow_reopen=False):
        dict_key = (attr_key, group_key)
        assert dict_key not in self.writers, "The group is already open for writing."

        if dict_key not in self.readers:
            reader = MsgpackReader(self._group_path(attr_key, group_key))
            self.readers[dict_key] = reader
        else:
            assert allow_reopen, "The group is already open for reading."

    def add(self, image_path, attr_key, group_key, image):
        dict_key = (attr_key, group_key)
        if dict_key not in self.writers:
            logging.debug(
                f"Adding {dict_key} for image {image_path} but no writer found, writers {self.writers}, ignoring."
            )
            return

        # This is to prevent mismatches in the image dict.
        new_image = {}
        # for debugging cache
        new_image["cache_metadata"] = dict_key
        # key for item in the group
        new_image["key"] = image_path

        new_image["tensors"] = [k for k, v in image.items() if isinstance(v, torch.Tensor)]

        for k in image.keys():
            new_image[k] = image[k] if k not in new_image["tensors"] else image[k].numpy()

        writer = self.writers[dict_key]
        writer.write(new_image)

    def remove_group(self, attr_key, group_key):
        # Not re-opening the group for writing: if a slice inside a group is corrupted, we remove the group (and skip generation this time) and regenerate the group cache the next time we access the group.
        dict_key = (attr_key, group_key)
        if dict_key in self.readers:
            self.readers[dict_key]._close()
            del self.readers[dict_key]
        if dict_key in self.writers:
            self.writers[dict_key].close()
            del self.writers[dict_key]

        try:
            os.remove(self._group_path(attr_key, group_key))
        # Don't raise error if file does not exist.
        except OSError:
            pass


class AbstractLoader(ABC):
    pad_token = IMG_PAD_TOKEN

    def __init__(self, cache_path, augmentations, args, compress=None):
        self.augmentations = augmentations
        self.args = args
        if cache_path is not None:
            self.use_cache = True
            self.cache = cache(
                cache_path,
                no_cache_misses_allowed=args.dataloader.no_cache_misses_allowed,
                extension=self.cached_extension,
                compress=compress,
            )
            self.split_augmentations = split_augmentations_by_cache(augmentations)
        else:
            logging.debug(
                "Cache is disabled. This will slow down the training process. (To enable cache, use dataset.cache_path)"
            )
            self.use_cache = False
            self.composed_all_augmentations = ComposeAug(augmentations)

    @abstractmethod
    def load_input(self, path, additional=None):
        pass

    @property
    @abstractmethod
    def cached_extension(self):
        pass

    def get_image(self, image_path, additional=None):
        """
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        """
        if not self.use_cache:
            image = self.load_input(image_path, additional)
            image = self.composed_all_augmentations(image)
            return image

        ## Load from partially cached
        for key, post_augmentations in self.split_augmentations:
            if self.cache.exists(image_path, key):
                try:
                    image = self.cache.get(image_path, key)
                    image = apply_augmentations_and_cache(
                        image,
                        additional,
                        image_path,
                        post_augmentations,
                        self.cache,
                        cache_full_size=self.args.dataset.cache_full_img,
                        cache_last_cachable_only=self.args.dataset.cache_last_cachable_only,
                        base_key=key,
                    )
                    return image
                except Exception as e:
                    print(e)
                    hashed_key = md5(image_path)
                    corrupted_file = self.cache._file_path(key, hashed_key)
                    warnings.warn(CORUPTED_FILE_ERR.format(sys.exc_info()[0], e, traceback.format_exc()))
                    self.cache.rem(image_path, key)
        ##  If load from cache fails, load from scratch
        all_augmentations = self.split_augmentations[-1][1]
        image = self.load_input(image_path, additional)
        if image is None:
            # print("returning none from dataloader for path", image_path)
            return None
        image = apply_augmentations_and_cache(
            image,
            additional,
            image_path,
            all_augmentations,
            self.cache,
            cache_full_size=self.args.dataset.cache_full_img,
            cache_last_cachable_only=self.args.dataset.cache_last_cachable_only,
            base_key=key,
        )
        return image


class AbstractGroupLoader(AbstractLoader):
    """AbstractLoader with GroupCache"""

    def __init__(self, cache_path, augmentations, group_key_name, args):
        self.augmentations = augmentations
        self.group_key_name = group_key_name
        self.args = args
        if cache_path is not None:
            self.use_cache = True
            self.cache = GroupCache(cache_path, extension=self.cached_extension)
            self.split_augmentations = split_augmentations_by_cache(augmentations)
        else:
            logging.debug(
                "Cache is disabled. This will slow down the training process. (To enable cache, use dataset.cache_path)"
            )
            self.use_cache = False
            self.composed_all_augmentations = ComposeAug(args, augmentations)

    def get_volume(self, image_paths, metadata=None, use_annotations=True):
        group_key = metadata[self.group_key_name].replace("/", "_")
        # just load the deterministic sample
        attr_key, post_augmentations = self.split_augmentations[0]
        input_dict = []
        for idx, image_path in enumerate(image_paths):
            if not self.cache.is_being_written(attr_key, group_key) and self.cache.exists_group(
                attr_key, group_key, image_path=image_path
            ):
                try:
                    if idx == 0:
                        self.cache.open_group_for_reading(attr_key=attr_key, group_key=group_key, allow_reopen=True)
                    image = self.cache.get(image_path, attr_key=attr_key, group_key=group_key)
                    image["mask"] = get_scaled_annotation_mask(metadata["annotations"][idx], self.args)
                    image["input"] = torch.tensor(image["input"], dtype=torch.float32).unsqueeze(0)
                    image["mask"] = torch.tensor(image["mask"], dtype=torch.float32).unsqueeze(0)
                    input_dict.append(image)
                except Exception as e:
                    warnings.warn("GOT ERROR")
        return input_dict

    def get_image(self, image_path, additional=None, first_image_in_group=False):
        """
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        first_image_in_group: the only one that allows creating a reader/writer. Other group items will not be read from cache or cached without an existing reader/writer in the case of data corruption and the subsequent file removal.
        """
        if not self.use_cache:
            image = self.load_input(image_path, additional)
            image = self.composed_all_augmentations(image)
            return image

        group_key = additional[self.group_key_name].replace("/", "_")
        ## Load from partially cached
        for attr_key, post_augmentations in self.split_augmentations:
            # If the group is being written, we should not read from it.
            if (not self.cache.is_being_written(attr_key, group_key)) and self.cache.exists_group(
                attr_key, group_key, image_path=image_path
            ):
                try:
                    if first_image_in_group:
                        self.cache.open_group_for_reading(attr_key=attr_key, group_key=group_key, allow_reopen=True)
                    image = self.cache.get(image_path, attr_key=attr_key, group_key=group_key)
                    image["mask"] = get_scaled_annotation_mask(additional["annotations"], self.args)
                    # Load from group cache and then apply the augmentations
                    image = apply_augmentations_and_group_cache(
                        image,
                        additional,
                        image_path,
                        post_augmentations,
                        self.cache,
                        cache_full_size=self.args.dataset.cache_full_img,
                        cache_last_cachable_only=self.args.dataset.cache_last_cachable_only,
                        base_key=attr_key,
                        group_key=group_key,
                    )
                    return image
                except Exception as e:
                    warnings.warn(CORUPTED_FILE_ERR.format(sys.exc_info()[0], e, traceback.format_exc()))
                    self.cache.remove_group(attr_key=attr_key, group_key=group_key)

        ##  If load from cache fails, load from scratch
        all_augmentations = self.split_augmentations[-1][1]
        image = self.load_input(image_path, additional)
        image = apply_augmentations_and_group_cache(
            image,
            additional,
            image_path,
            all_augmentations,
            self.cache,
            cache_full_size=self.args.dataset.cache_full_img,
            cache_last_cachable_only=self.args.dataset.cache_last_cachable_only,
            base_key=attr_key,
            group_key=group_key,
            open_writer=first_image_in_group,
        )
        return image


def apply_augmentations_and_group_cache(
    image,
    additional,
    img_path,
    augmentations,
    cache,
    cache_full_size=False,
    cache_last_cachable_only=False,
    base_key="",
    group_key="",
    open_writer=False,
):
    """
    Loads the image by its absolute path and apply the augmentations one
    by one (similar to what the composed one is doing).  All first cachable
    transformer's output is cached (until reaching a non cachable one).
    """
    if cache_full_size and not cache.exists(img_path, DEFAULT_CACHE_DIR):
        if open_writer:
            cache.open_group_for_writing(attr_key=DEFAULT_CACHE_DIR, group_key=group_key)
        cache.add(img_path, attr_key=DEFAULT_CACHE_DIR, group_key=group_key, image=image)

    all_prev_cachable = True
    key = base_key

    if len(augmentations):
        last_cachable_index = (
            min(
                [ind for ind, augmentation in enumerate(augmentations) if not augmentation.cachable()],
                default=len(augmentations),
            )
            - 1
        )

        for ind, trans in enumerate(augmentations):
            if additional is not None:
                image = trans(image, **additional)
            else:
                image = trans(image)
            if not all_prev_cachable or not trans.cachable():
                all_prev_cachable = False
            else:
                key += trans.caching_keys()
                if (not cache_last_cachable_only) or (ind == last_cachable_index):
                    if open_writer:
                        cache.open_group_for_writing(attr_key=key, group_key=group_key)
                    cache.add(img_path, attr_key=key, group_key=group_key, image=image)

    return image
