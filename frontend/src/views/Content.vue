<template>
  <div id="textArea">
    <!-- contents -->
    <div id="Posts">
         <Post />
    </div>
  </div>
</template>

<script>
import Vue from 'vue'
import VueRouter from 'vue-router'
Vue.use(VueRouter)

  import Post from '../components/layout/Post.vue'

  export default Vue.extend({
		data: () => ({

		}),
		name: "Header",
		components:{
			Post,
		},
	})

</script>
<style>

</style>